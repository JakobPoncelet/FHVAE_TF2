import os
import sys
import time
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

    optimizer = tf.keras.optimizers.Adam(learning_rate=conf['lr'], beta_1=conf['beta1'], beta_2=conf['beta2'], epsilon=conf['adam_eps'], amsgrad=False)
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    checkpoint_directory = os.path.join(exp_dir, 'training_checkpoints')
    os.makedirs(checkpoint_directory, exist_ok=True)
    #checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_directory, max_to_keep=5)

    logdir = os.path.join(exp_dir, "logdir")
    os.makedirs(logdir, exist_ok=True)
    writer = tf.summary.create_file_writer(logdir)
    writer.set_as_default()

    step_losses = []
    mean_losses = []

    meanloss_file = os.path.join(exp_dir, 'result_mean_loss.txt')
    comploss_file = os.path.join(exp_dir, 'result_comp_loss.txt')

    if os.path.exists(meanloss_file):
        os.remove(meanloss_file)
    if os.path.exists(comploss_file):
        os.remove(comploss_file)

    flag = True

    for epoch in range(conf['n_epochs']):
        print("EPOCH %i\n" % (epoch+1))
        start = time.time()
        train_loss.reset_states()

        s_seqs = sample_tr_seqs(conf['nmu2'])
        s_iterator = lambda: tr_iterator_by_seqs(s_seqs, seg_rem=True)

        # estimate and update mu2 lookup table
        mu2_dict = estimate_mu2_dict(model, s_iterator)
        mu2_table = np.array([mu2_dict[idx] for idx in range(len(mu2_dict))])
        model.mu2_table.assign(mu2_table)

        # train loop over dataset
        for xval, yval, nval, cval in tr_iterator_by_seqs(s_seqs):

            # FOR NOW USE CVAL FOR BOTH bREG AND cREG....
            step_loss, log_pmu2, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy, log_b_loss, log_c_loss, flag \
                = train_step(model, tf.stack(xval, axis=0), yval, nval, cval, cval, optimizer, train_loss, flag)

            step_losses.append(step_loss)

            # print the results to a file
            with open(comploss_file, "a+") as pid:
                pid.write("Loss: %f \t \t lb=%f \t log_qy=%f \t log_b=%f \t log_c=%f \n" % \
                 (step_loss, -tf.reduce_mean(lb), -tf.reduce_mean(log_qy), -tf.reduce_mean(log_b_loss), -tf.reduce_mean(log_c_loss)))
                pid.write("\t lower bound components: \t log_pmu2=%f \t neg_kld_z2=%f \t neg_kld_z1=%f \t log_px_z=%f \n" % \
                 (-tf.reduce_mean(log_pmu2), -tf.reduce_mean(neg_kld_z2), -tf.reduce_mean(neg_kld_z1), -tf.reduce_mean(log_px_z)))

        print('Resulting mean-loss of epoch {} is {:.4f}, which took {} seconds to run'.format(epoch+1, train_loss.result(), time.time()-start))
        mean_losses.append(float(train_loss.result()))

        with open(meanloss_file, "a+") as fid:
            fid.write('Resulting mean-loss of epoch {} is {:.4f}, which took {} seconds to run\n'.format(epoch+1, train_loss.result(), time.time()-start))


        tf.summary.scalar('train_loss', train_loss.result(), step=epoch)

        #checkpoint.save(file_prefix=checkpoint_prefix)
        manager.save()

    plt.figure('result-steploss')
    plt.plot(step_losses)
    plt.xlabel('Batch #')
    plt.ylabel('Step Loss')
    plt.savefig(os.path.join(exp_dir, 'result_comp_loss.pdf'), format='pdf')

    plt.figure('result-meanloss')
    plt.plot(mean_losses)
    plt.xlabel('Epochs #')
    plt.ylabel('Mean Loss')
    plt.savefig(os.path.join(exp_dir, 'result_mean_loss.pdf'), format='pdf')

#@tf.function
def train_step(model, x, y, n, bReg, cReg, optimizer, train_loss, flag):
    """
    train fhvae step by step and compute the gradients from the losses
    """

    with tf.GradientTape() as tape:
        mu2, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits = model(x, y)

        loss, log_pmu2, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy, log_b_loss, log_c_loss = \
            model.compute_loss(x, y, n, bReg, cReg, mu2, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits)

    print("Loss: %f \t \t lb=%f \t log_qy=%f \t log_b=%f \t log_c=%f" % \
          (loss, -tf.reduce_mean(lb), -tf.reduce_mean(log_qy), -tf.reduce_mean(log_b_loss), -tf.reduce_mean(log_c_loss)))

    print("\t lower bound components: \t log_pmu2=%f \t neg_kld_z2=%f \t neg_kld_z1=%f \t log_px_z=%f" % \
          (-tf.reduce_mean(log_pmu2), -tf.reduce_mean(neg_kld_z2), -tf.reduce_mean(neg_kld_z1), -tf.reduce_mean(log_px_z)))

    # apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if flag:
        for v in model.trainable_variables:
            print((v.name, v.shape))
        flag = False


    # update keras mean loss metric
    train_loss(loss)

    return loss, log_pmu2, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy, log_b_loss, log_c_loss, flag

#@tf.function
def estimate_mu2_dict(model, iterator):
    """
    estimate mu2 for sequences produced by iterator
    Args: model(FHVAE):
          iterator(Callable):
    Return: mu2_dict(dict): sequence index to mu2 dict
    """

    nseg_table = defaultdict(float)
    z2_sum_table = defaultdict(float)

    for x_val, y_val, _, _ in iterator():
        z2_mu = model.encoder.z2_mu_separate(tf.stack(x_val, axis=0))


        for idx, _y in enumerate(y_val):
            z2_sum_table[_y] += z2_mu[idx, :]
            nseg_table[_y] += 1

        # the above instead of the zip-method below because TF2 does not allow iterating over tensors

        #for _y, _z2 in zip(y_val, z2_mu):
        #    z2_sum_table[_y] += _z2
        #    nseg_table[_y] += 1

    mu2_dict = dict()
    for _y in nseg_table:
        z2_sum = z2_sum_table[_y]
        n = nseg_table[_y]
        r = np.exp(model.pz2_stddev ** 2) / np.exp(model.pmu2_stddev ** 2)
        mu2_dict[_y] = z2_sum / (n+r)

    return mu2_dict


# while True:
#     s_seqs = sample_tr_seqs(nmu2)
#     s_iterator = lambda: tr_iterator_by_seqs(s_seqs, seg_rem=True)
#     mu2_dict = _est_mu2_dict(sess, model, s_iterator)
#     mu2_table = np.array([mu2_dict[idx] for idx in range(len(mu2_dict))])
#     update_mu2_table(sess, model, mu2_table)
#     for x_val, y_val, n_val, c_val in tr_iterator_by_seqs(s_seqs):
#         feed_dict = _feed_dict(x_val, y_val, n_val, c_val)
#         global_step, _ = sess.run([global_step_var, apply_grad_op], feed_dict)
#
#         if global_step % n_print_steps == 0 and global_step != init_step:
#             feed_dict = _feed_dict(x_val, y_val, n_val, c_val)
#             tr_sum_vals = sess.run(tr_sum_vars, feed_dict)
#             is_diverged = _print_prog(tr_sum_names, tr_sum_vals)
#             if is_diverged:
#                 print("training diverged...")
#                 return
#             ptime = time.time()
#
#         if global_step % n_steps_per_epoch == 0 and global_step != init_step:
#             is_best, dt_sum_vals = _valid_step()
#             if is_best:
#                 best_epoch, best_dt_lb = epoch, dt_sum_vals[0]
#             saver.save(sess, "%s/models/fhvae" % exp_dir, global_step=global_step)
#             _save_prog(dt_sum_vals)
#             epoch += 1
#             if _check_terminate(epoch, best_epoch, n_patience, n_epochs):
#                 print("training finished...")
#                 return
#             etime = time.time()
#     passes += 1

