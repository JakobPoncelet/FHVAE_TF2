from __future__ import division
import os
import sys
import time
import math
import numpy as np
import tensorflow as tf
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def hs_train_reg(exp_dir, model, conf, sample_tr_seqs, tr_iterator_by_seqs, dt_iterator):
    """
    train fhvae with hierarchical sampling
    """
    if conf['lr'] == 'custom':
        learning_rate = CustomSchedule(conf['d_model'], warmup_steps=conf['warmup_steps'])
    else:
        learning_rate = conf['lr']

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=conf['beta1'], beta_2=conf['beta2'], epsilon=conf['adam_eps'], amsgrad=False)
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    checkpoint_directory = os.path.join(exp_dir, 'training_checkpoints')
    os.makedirs(checkpoint_directory, exist_ok=True)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_directory, max_to_keep=conf['n_patience']+1)

    logdir = os.path.join(exp_dir, "logdir")
    os.makedirs(logdir, exist_ok=True)
    writer = tf.summary.create_file_writer(logdir)
    writer.set_as_default()

    mean_losses = []
    valid_losses = []

    # print all losses to text files
    lossdir = os.path.join(exp_dir, "losses")
    os.makedirs(lossdir, exist_ok=True)
    meanloss_file = os.path.join(lossdir, 'result_mean_loss.txt')
    validloss_file = os.path.join(lossdir, 'result_valid_loss.txt')
    comploss_file = os.path.join(lossdir, 'result_comp_loss_0.txt')

    if os.path.exists(meanloss_file):
        os.remove(meanloss_file)
    if os.path.exists(comploss_file):
        os.remove(comploss_file)
    if os.path.exists(validloss_file):
        os.remove(validloss_file)

    flag = True

    best_epoch, best_valid_loss = 0, np.inf
    start = time.time()

    for epoch in range(1, conf['n_epochs']+1):
        print("EPOCH %i\n" % epoch)
        epoch_start = time.time()
        train_loss.reset_states()

        # new file every 50 epochs (becomes too large otherwise)
        comploss_file = os.path.join(lossdir, 'result_comp_loss_%s.txt') % (str(int(epoch/50)))

        # hierarchical sampling
        s_seqs = sample_tr_seqs(conf['nmu2'])
        s_iterator = lambda: tr_iterator_by_seqs(s_seqs, bs=conf['batch_size'], seg_rem=True)

        # estimate and update mu2 lookup table
        mu2_dict = estimate_mu2_dict(model, s_iterator)
        mu2_table = np.array([mu2_dict[idx] for idx in range(len(mu2_dict))])
        model.mu2_table.assign(mu2_table)

        # train loop over the samples from the dataset
        for xval, yval, nval, cval, bval in tr_iterator_by_seqs(s_seqs, bs=conf['batch_size']):

            # CORE TRAINING STEP
            step_loss, log_pmu2, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy, log_b_loss, log_c_loss, flag \
                = train_step(model, tf.stack(tf.cast(xval, dtype=tf.float32), axis=0), yval, nval, bval, cval, optimizer, train_loss, flag, epoch)

            # print the results to a file
            with open(comploss_file, "a+") as pid:
                pid.write("Loss: %f \t \t lb=%f \t log_qy=%f \t log_b=%f \t log_c=%f \n" % \
                 (step_loss, -tf.reduce_mean(lb), -tf.reduce_mean(log_qy), -tf.reduce_mean(log_b_loss), -tf.reduce_mean(log_c_loss)))
                pid.write("\t lower bound components: \t log_pmu2=%f \t neg_kld_z2=%f \t neg_kld_z1=%f \t log_px_z=%f \n" % \
                 (-tf.reduce_mean(log_pmu2), -tf.reduce_mean(neg_kld_z2), -tf.reduce_mean(neg_kld_z1), -tf.reduce_mean(log_px_z)))

        print('Resulting mean-loss of epoch {} is {:.4f}, which took {} seconds to run'.format(epoch, train_loss.result(), time.time()-epoch_start))
        mean_losses.append(float(train_loss.result()))
        with open(meanloss_file, "a+") as fid:
            fid.write('Resulting mean-loss of epoch {} is {:.4f}, which took {} seconds to run\n'.format(epoch, train_loss.result(), time.time()-epoch_start))

        tf.summary.scalar('train_loss', train_loss.result(), step=epoch)

        manager.save()

        # validation step
        valid_loss, normalloss = validation_step(model, dt_iterator, conf)

        valid_losses.append(valid_loss)
        print('Validation loss of epoch {} is {:.4f}, and {:.4f} when calculated differently \n'.format(epoch, valid_loss, normalloss))
        with open(validloss_file, "a+") as fid:
            fid.write('Validation loss of epoch {} is {:.4f}, and {:.4f} when calculated differently \n'.format(epoch, valid_loss, normalloss))

        # early stopping
        best_epoch, best_valid_loss, is_finished = check_finished(conf, epoch, best_epoch, valid_loss, best_valid_loss)
        if is_finished:
            with open(os.path.join(checkpoint_directory, 'best_checkpoint'), 'w+') as lid:
                lid.write(best_epoch)
            break

    print('Complete run over {} epochs took {} seconds\n'.format(conf['n_epochs'], time.time()-start))
    print('Best run was in epoch {} with a validation loss of {} \n'.format(best_epoch, best_valid_loss))

    # make plots of loss after training
    plt.figure('result-meanloss')
    plt.plot(mean_losses)
    plt.xlabel('Epochs #')
    plt.ylabel('Mean Loss')
    plt.savefig(os.path.join(exp_dir, 'result_mean_loss.pdf'), format='pdf')

    plt.figure('result-validloss')
    plt.plot(valid_losses)
    plt.xlabel('Epochs #')
    plt.ylabel('Mean Loss')
    plt.savefig(os.path.join(exp_dir, 'result_valid_loss.pdf'), format='pdf')


#@tf.function
def train_step(model, x, y, n, bReg, cReg, optimizer, train_loss, flag, epoch):
    """
    train fhvae step by step and compute the gradients from the losses
    """

    with tf.GradientTape() as tape:

        mu2, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits = model(x, y)

        loss, log_pmu2, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy, log_b_loss, log_c_loss = \
            model.compute_loss(x, y, n, bReg, cReg, mu2, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits)

    # apply gradients
    gradients = tape.gradient(loss, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.NONE)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # print the variables once at the beginning of training
    if flag:
        for v in model.trainable_variables:
            print((v.name, v.shape))
        flag = False

    # print separate losses in terminal during first epochs for debugging purposes
    if epoch < 3:
        print("Loss: %f \t \t lb=%f \t log_qy=%f \t log_b=%f \t log_c=%f" % \
              (loss, -tf.reduce_mean(lb), -tf.reduce_mean(log_qy), -tf.reduce_mean(log_b_loss), -tf.reduce_mean(log_c_loss)))
        print("\t lower bound components: \t log_pmu2=%f \t neg_kld_z2=%f \t neg_kld_z1=%f \t log_px_z=%f" % \
              (-tf.reduce_mean(log_pmu2), -tf.reduce_mean(neg_kld_z2), -tf.reduce_mean(neg_kld_z1), -tf.reduce_mean(log_px_z)))

    # update keras mean loss metric
    train_loss(loss)

    return loss, log_pmu2, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy, log_b_loss, log_c_loss, flag


#@tf.function
def estimate_mu2_dict(model, iterator, bs=256, validation=False):
    """
    estimate mu2 for sequences produced by iterator
    Args: model(FHVAE):
          iterator(Callable):
    Return: mu2_dict(dict): sequence index to mu2 dict
    """

    nseg_table = defaultdict(float)
    z2_sum_table = defaultdict(float)

    kwargs = {}
    if validation:
        kwargs = {'bs':bs}

    # calculate sum over all z2_mu's
    for x_val, y_val, _, _, _ in iterator(**kwargs):
        z2_mu = model.encoder.z2_mu_separate(tf.stack(tf.cast(x_val, dtype=tf.float32), axis=0))

        for idx, _y in enumerate(y_val):
            z2_sum_table[_y] += z2_mu[idx, :]
            nseg_table[_y] += 1

    # mu2-formula from paper
    mu2_dict = dict()
    for _y in nseg_table:
        z2_sum = z2_sum_table[_y]
        n = nseg_table[_y]
        r = np.exp(model.pz2_stddev ** 2) / np.exp(model.pmu2_stddev ** 2)
        mu2_dict[_y] = z2_sum / (n+r)

    return mu2_dict


def validation_step(model, dt_iterator, conf):
    """
    calculate loss on development set
    """
    validloss = 0.0
    tot_segs = 0.0
    normalloss = 0.0

    mu2_dict = estimate_mu2_dict(model, dt_iterator, bs=conf['batch_size'], validation=True)
    mu2_table = np.array([mu2_dict[idx] for idx in range(len(mu2_dict))])

    # pad to size=nmu2 as initialized in model (however will not work if development set is larger than nmu2.....)
    filler = np.zeros(((int(conf['nmu2']) - mu2_table.shape[0]), mu2_table.shape[1]))
    mu2_table = np.vstack((mu2_table, filler))
    model.mu2_table.assign(mu2_table)

    for xval, yval, nval, cval, bval in dt_iterator(bs=conf['batch_size']):

        mu2, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits = model(tf.stack(tf.cast(xval, dtype=tf.float32), axis=0), yval)

        loss, log_pmu2, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy, log_b_loss, log_c_loss = \
            model.compute_loss(tf.stack(tf.cast(xval, dtype=tf.float32), axis=0), yval, nval, bval, cval, mu2, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits,
                               z2_rlogits)

        validloss += loss * len(xval)
        tot_segs += len(xval)
        normalloss += loss

    return validloss / tot_segs, normalloss


def check_finished(conf, epoch, best_epoch, val_loss, best_val_loss):
    """
    stop if validation loss doesnt improve after n_patience epochs
    """
    is_finished = False
    if val_loss < best_val_loss:
        best_epoch = epoch
        best_val_loss = val_loss

    if (best_epoch - epoch) > conf['n_patience']:
        is_finished = True

    if math.isnan(val_loss):
        is_finished = True

    return best_epoch, best_val_loss, is_finished


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    # define custom learning rate schedule as according to transformer paper
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
