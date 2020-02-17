from __future__ import division
import os
import sys
import time
import pickle
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
from sklearn.manifold import TSNE
import tensorflow as tf
from .plotter import plot_x, scatter_plot

np.random.seed(123)

def test_reg(expdir, model, conf, dt_iterator, dt_iterator_by_seqs, dt_seqs, dt_dset):
    '''
    Compute variational lower bound
    '''

    print("\nRESTORING MODEL")
    starttime = time.time()
    optimizer = tf.keras.optimizers.Adam(learning_rate=conf['lr'], beta_1=conf['beta1'], beta_2=conf['beta2'], epsilon=conf['adam_eps'], amsgrad=False)
    checkpoint_directory = os.path.join(expdir, 'training_checkpoints')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    if os.path.exists(os.path.join(checkpoint_directory, 'best_checkpoint')):
        with open(os.path.join(checkpoint_directory, 'best_checkpoint'), 'r') as pid:
            best_checkpoint = (pid.readline()).rstrip()
        status = checkpoint.restore(os.path.join(checkpoint_directory, 'ckpt-' + str(best_checkpoint)))
    else:
        manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_directory, max_to_keep=5)
        status = checkpoint.restore(manager.latest_checkpoint)

    print("restoring model takes %.2f seconds" % (time.time()-starttime))
    status.assert_existing_objects_matched()
    #status.assert_consumed()

    os.makedirs(os.path.join(expdir, 'test', 'img', 'x_tra'), exist_ok=True)
    os.makedirs(os.path.join(expdir, 'test', 'spec'), exist_ok=True)
    os.makedirs(os.path.join(expdir, 'test', 'txt'), exist_ok=True)

    print("\nCOMPUTING AVERAGE VALUES")
    avg_loss, avg_vals = compute_average_values(model, dt_iterator, conf)

    print("\nCOMPUTING VALUES BY SEQUENCE")
    z1_by_seq, z2_by_seq, mu2_by_seq, regpost_by_seq, xin_by_seq, xout_by_seq, xoutv_by_seq, z1reg_by_seq, regpost_by_seq_z1, \
        z2reg_by_seq, bReg_by_seq, cReg_by_seq = compute_values_by_seq(model, conf, dt_iterator_by_seqs, dt_seqs, expdir)

    print("\nCOMPUTING PREDICTION ACCURACIES FOR LABELS FROM Z2")
    labels, accs = compute_pred_acc_z2(expdir, model, conf, dt_seqs, dt_dset, regpost_by_seq, z2reg_by_seq, cReg_by_seq)

    print("\nCOMPUTING PREDICTION ACCURACIES FOR TIME ALIGNED LABELS FROM Z1")
    labels, accs = compute_pred_acc_z1(expdir, model, conf, dt_seqs, dt_dset, regpost_by_seq, z1reg_by_seq, bReg_by_seq)

    print("\nVISUALIZING RESULTS")
    visualize_reg_vals(expdir, model, dt_seqs, conf, z1_by_seq, z2_by_seq, mu2_by_seq, regpost_by_seq, xin_by_seq, \
                       xout_by_seq, xoutv_by_seq, z1reg_by_seq)

    print("\nVISUALIZING TSNE BY LABEL")
    tsne_by_label(expdir, model, conf, dt_iterator_by_seqs, dt_seqs, dt_dset)

    print("\nFINISHED\nResults stored in %s/test" % expdir)


def compute_average_values(model, dt_iterator, conf):

    mu2_dict = estimate_mu2_dict(model, conf, dt_iterator)
    _print_mu2_stat(mu2_dict)
    mu2_table = np.array([mu2_dict[idx] for idx in range(len(mu2_dict))])
    # pad to size=nmu2 as initialized in model (however will not work if test set is larger than nmu2.....)
    filler = np.zeros(((int(conf['nmu2'])-mu2_table.shape[0]), mu2_table.shape[1]))
    mu2_table = np.vstack((mu2_table, filler))
    model.mu2_table.assign(mu2_table)

    sum_names = ['log_pmu2', 'neg_kld_z2', 'neg_kld_z1', 'log_px_z', 'lb', 'log_qy', 'log_b_loss', 'log_c_loss']
    sum_loss = 0.
    sum_vals = [0. for _ in range(len(sum_names))]
    tot_segs = 0.
    avg_vals = [0. for _ in range(len(sum_names))]

    for x_val, y_val, n_val, c_val, b_val in dt_iterator(bs=conf['batch_size']):
        x_val = tf.stack(tf.cast(x_val, dtype=tf.float32), axis=0)

        mu2, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits = model(x_val, y_val)

        loss, log_pmu2, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy, log_b_loss, log_c_loss = \
            model.compute_loss(x_val, y_val, n_val, b_val, c_val, mu2, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits)

        results = [log_pmu2, neg_kld_z2, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy, log_b_loss, log_c_loss]
        for i in range(len(sum_vals)):
            sum_vals[i] += -tf.reduce_mean(results[i])
        sum_loss += loss
        tot_segs += len(x_val)

    avg_loss = sum_loss/tot_segs
    print("average loss = %f" % (avg_loss))
    for i in range(len(sum_vals)):
        avg_vals[i] = sum_vals[i]/tot_segs
        print("\t average value for %s \t= %f" % (sum_names[i], avg_vals[i]))

    return avg_loss, avg_vals


def compute_values_by_seq(model, conf, iterator_by_seqs, seqs, expdir):

    z1_by_seq = defaultdict(list)
    z2_by_seq = defaultdict(list)
    mu2_by_seq = dict()
    regpost_by_seq = dict()
    xin_by_seq = defaultdict(list)
    xout_by_seq = defaultdict(list)
    xoutv_by_seq = defaultdict(list)
    z1reg_by_seq = defaultdict(list)
    z2reg_by_seq = defaultdict(list)
    regpost_by_seq_z1 = dict()
    bReg_by_seq = defaultdict(list)
    cReg_by_seq = defaultdict(list)

    for seq in seqs:
        for x, _, _, c, b in iterator_by_seqs([seq], bs=conf['batch_size']):

            x = tf.stack(tf.cast(x, dtype=tf.float32), axis=0)

            _, _, _, _, _, _, qz1_x, qz2_x = model.encoder(x)
            z2_by_seq[seq].append(qz2_x[0])
            z1_by_seq[seq].append(qz1_x[0])

            xin_by_seq[seq].append(x)

            _, _, _, _, px_z = model.decoder(x, qz1_x[0], qz2_x[0])
            xout_by_seq[seq].append(px_z[0])
            xoutv_by_seq[seq].append(px_z[1])

            # probabilities of each of the regularisation classes given mean(z1)
            z1_rlogits, z2_rlogits = model.regulariser(qz1_x[0], qz2_x[0])
            z1reg_by_seq[seq] = list(map(_softmax, z1_rlogits))
            z2reg_by_seq[seq] = list(map(_softmax, z2_rlogits))

            cReg_by_seq[seq].append(c)
            bReg_by_seq[seq].append(b)

        z1_by_seq[seq] = np.concatenate(z1_by_seq[seq], axis=0)
        z2_by_seq[seq] = np.concatenate(z2_by_seq[seq], axis=0)
        xin_by_seq[seq] = np.concatenate(xin_by_seq[seq], axis=0)
        xout_by_seq[seq] = np.concatenate(xout_by_seq[seq], axis=0)
        xoutv_by_seq[seq] = np.concatenate(xoutv_by_seq[seq], axis=0)
        # z1reg_by_seq[seq] = np.concatenate(z1reg_by_seq[seq], axis=0)

        bReg_by_seq[seq] = np.concatenate(bReg_by_seq[seq], axis=0)
        cReg_by_seq[seq] = np.concatenate(cReg_by_seq[seq], axis=0)


        # formula for inferring S-vector mu2 during testing, paper p5 (over all segments from same sequence)
        z2_sum = np.sum(z2_by_seq[seq], axis=0)
        n = len(z2_by_seq[seq])
        r = np.exp(model.pz2_stddev ** 2) / np.exp(model.pmu2_stddev ** 2)
        mu2_by_seq[seq] = z2_sum / (n+r)

        d2 = mu2_by_seq[seq].shape[0]
        z2 = np.asarray(mu2_by_seq[seq]).reshape([1, d2])

        # probabilities of each of the regularisation classes given the computed z2 of above
        _, z2_rlogits = model.regulariser(z1_by_seq[seq], z2)
        regpost_by_seq[seq] = list(map(_softmax, z2_rlogits))

        # formula for inferring alternative S-vector mu1 during testing, paper p7
        z1_sum = np.sum(z1_by_seq[seq], axis=0)
        n = len(z1_by_seq[seq])
        r = np.exp(model.pz1_stddev ** 2)
        z1 = z1_sum / (n+r)
        z1 = np.asarray(z1).reshape([1, z1.shape[0]])

        # probabilities given computed mu1
        z1_rlogits, _ = model.regulariser(z1, z2_by_seq[seq])
        regpost_by_seq_z1[seq] = list(map(_softmax, z1_rlogits))

    # # save the mu2
    # with open(os.path.join(expdir, 'test', 'mu2_by_seq.txt'),"w"):
    #     for seq in seqs:
    #         f.write( ' '.join (map(str,mu2_by_seq[seq])) )
    #         f.write('\n')

    # save the mean mu2
    if not os.path.exists(os.path.join(expdir, 'test', 'mu2_by_seq.npy')):
        mumu = np.zeros([mu2_by_seq[seqs[1]].size])
        for seq in seqs:
            mumu += mu2_by_seq[seq]
        mumu /= len(seqs)
        with open(os.path.join(expdir, 'test', 'mu2_by_seq.npy'), "wb") as fnp:
            np.save(fnp, mumu)

    return z1_by_seq, z2_by_seq, mu2_by_seq, regpost_by_seq, xin_by_seq, xout_by_seq, xoutv_by_seq, z1reg_by_seq, regpost_by_seq_z1, z2reg_by_seq, bReg_by_seq, cReg_by_seq


def compute_pred_acc_z2(expdir, model, conf, seqs, dt_dset, regpost_by_seq, z2reg_by_seq, cReg_by_seq):

    names = conf['facs'].split(':')
    lab2idx= conf['lab2idx']
    accuracies = [0. for _ in range(len(names))]
    #accuracies_z2reg = [0. for _ in range(len(names))]

    for i, name in enumerate(names):

        ordered_labs = lab2idx[name]
        truelabs = dt_dset.labs_d[name].seq2lab

        total = 0

        correct = 0  #using mu2
        #correct_z2reg = 0  #using z2_rlogits

        with open("%s/test/txt/%s_predictions.scp" % (expdir, name), "w") as f:
            f.write("#seq truelabel predictedlabel      for class %s \n" % name)
            for seq in seqs:
                total += 1
                probs = regpost_by_seq[seq][i]
                # the predicted labels are shifted by one for some reason, empty label is now at end instead of start
                pred_lab = ordered_labs[np.argmax(probs)]
                if pred_lab == truelabs[seq]:
                    correct += 1
                f.write(seq+" "+str(truelabs[seq])+" "+str(pred_lab)+"\n")

                #probs_z2reg = z2reg_by_seq[seq][i]
                #pred_lab_z2reg = ordered_labs[np.argmax(np.sum(probs_z2reg, axis=0))+1]
                #if pred_lab_z2reg == truelabs[seq]:
                #    correct_z2reg += 1

        accuracies[i] = correct/total
        with open("%s/test/txt/%s_acc" % (expdir, name), "w") as fid:
            fid.write("%10.3f \n" % accuracies[i])
        print("prediction accuracy for labels of class %s is %f" % (name, accuracies[i]))

        #accuracies_z2reg[i] = correct_z2reg/total
        #print("prediction accuracy for labels of class from z2reg_by_seq %s is %f" % (name, accuracies[i]))

    return names, accuracies


def compute_pred_acc_z1(expdir, model, conf, seqs, dt_dset, regpost_by_seq, z1reg_by_seq, bReg_by_seq):

    names = conf['talabs'].split(':')

    accuracies = [0. for _ in range(len(names))]

    for i, name in enumerate(names):
        with open("%s/test/txt/%s_predictions.scp" % (expdir, name), "w") as f:
            f.write("#segmentnumber true prediction \n")
            total = 0
            correct = 0

            for seq in seqs:
                nsegs = z1reg_by_seq[seq][0].shape[0]
                f.write('Sequence %s with %i segments \n' % (seq, nsegs))

                for j in range(nsegs):
                    total += 1

                    truelab = bReg_by_seq[seq][j, i]
                    pred_lab = np.argmax(z1reg_by_seq[seq][i][j, :])

                    if pred_lab == truelab:
                        correct += 1

                    f.write("\t %i \t %i \t %i \n" % (j, truelab, pred_lab))

        accuracies[i] = correct/total
        with open("%s/test/txt/%s_acc" % (expdir, name), "w") as fid:
            fid.write("%10.3f \n" % accuracies[i])
        print("prediction accuracy for labels of class %s is %f" % (name, accuracies[i]))

    return names, accuracies


def visualize_reg_vals(expdir, model, seqs, conf, z1_by_seq, z2_by_seq, mu2_by_seq, regpost_by_seq, xin_by_seq, xout_by_seq, xoutv_by_seq, z1reg_by_seq):

    if True:
        # names = ["region", "gender"]
        names = conf['facs'].split(':')
        for i, name in enumerate(names):
            with open("%s/test/txt/%s.scp" % (expdir, name), "w") as f:
                for seq in seqs:
                    f.write(seq + "  [ ")
                    for e in np.nditer(regpost_by_seq[seq][i]):
                        f.write("%10.3f " % e)
                    f.write("]\n")

    if True:
        names = ["pho"]
        #names = conf['talabs']  #.split(':')
        for i, name in enumerate(names):
            os.makedirs("%s/test/txt/%s" % (expdir, name), exist_ok=True)
            for seq in seqs:
                np.save("%s/test/txt/%s/%s" % (expdir, name, seq), z1reg_by_seq[seq][i])

    print("only using 10 random sequences for visualization")
    seqs = sorted(list(np.random.choice(seqs, 10, replace=False)))
    seq_names = ["%02d_%s" % (i, seq) for i, seq in enumerate(seqs)]

    if True:
        # visualize reconstruction
        print("visualizing reconstruction")
        plot_x([xin_by_seq[seq] for seq in seqs], seq_names, "%s/test/img/xin.png" % expdir)
        plot_x([xout_by_seq[seq] for seq in seqs], seq_names, "%s/test/img/xout.png" % expdir)
        plot_x([xoutv_by_seq[seq] for seq in seqs], seq_names,
               "%s/test/img/xout_logvar.png" % expdir, clim=(None, None))

    if True:
        # factorization: use the centered segment from each sequence
        print("visualizing factorization")
        cen_z1 = np.array([z1_by_seq[seq][(np.floor(len(z1_by_seq[seq])/2)).astype(int), :] for seq in seqs])
        cen_z2 = np.array([z2_by_seq[seq][(np.floor(len(z2_by_seq[seq])/2)).astype(int), :] for seq in seqs])
        xfac = []
        for z1 in cen_z1:
            z1 = np.tile(z1, (len(cen_z2), 1))
            _, _, _, _, px_z = model.decoder(\
                np.zeros((len(z1), conf['tr_shape'][0], conf['tr_shape'][1]), dtype=np.float32), z1, cen_z2)
            xfac.append(px_z[0])
        plot_x(xfac, seq_names, "%s/test/img/xfac.png" % expdir, sep=True)

    if True:
        # Maybe use dataset instead of dataset_test?
        with open(os.path.join(conf['datadir'], conf['dataset'], 'train', 'mvn.pkl'), "rb") as f:
            mvn_params = pickle.load(f)
        nb_mel = mvn_params["mean"].size
        for src_seq, src_seq_name in zip(seqs, seq_names):
            with open("%s/test/spec/xin_%s.npy" % (expdir, src_seq), "wb") as fnp:
                np.save(fnp, np.reshape(xin_by_seq[src_seq], (-1, nb_mel)) * mvn_params["std"] + mvn_params["mean"])
            with open("%s/test/spec/xout_%s.npy" % (expdir, src_seq), "wb") as fnp:
                np.save(fnp,
                        np.reshape(xout_by_seq[src_seq], (-1, nb_mel)) * mvn_params["std"] + mvn_params["mean"])

    if True:
        # sequence neutralisation
        print("visualizing neutral sequences")
        neu_by_seq = dict()
        with open("%s/test/mu2_by_seq.npy" % expdir, "rb") as fnp:
            mumu = np.float32(np.load(fnp))
        for src_seq, src_seq_name in zip(seqs, seq_names):
            del_mu2 = mumu - mu2_by_seq[src_seq]
            src_z1, src_z2 = z1_by_seq[src_seq], z2_by_seq[src_seq]
            neu_by_seq[src_seq] = _seq_translate(
                model, conf['tr_shape'], src_z1, src_z2, del_mu2)
            with open("%s/test/spec/neu_%s.npy" % (expdir, src_seq), "wb") as fnp:
                np.save(fnp, np.reshape(neu_by_seq[src_seq], (-1, nb_mel)) * mvn_params["std"] + mvn_params["mean"])

        plot_x([neu_by_seq[seq] for seq in seqs], seq_names,
               "%s/test/img/neutral.png" % expdir, False)

    if True:
        # sequence translation
        print("visualizing sequence translation")
        xtra_by_seq = dict()
        for src_seq, src_seq_name in zip(seqs, seq_names):
            xtra_by_seq[src_seq] = dict()
            src_z1, src_z2 = z1_by_seq[src_seq], z2_by_seq[src_seq]
            for tar_seq in seqs:
                del_mu2 = mu2_by_seq[tar_seq] - mu2_by_seq[src_seq]
                xtra_by_seq[src_seq][tar_seq] = _seq_translate(
                    model, conf['tr_shape'], src_z1, src_z2, del_mu2)
                with open("%s/test/spec/src_%s_tar_%s.npy" % (expdir, src_seq, tar_seq), "wb") as fnp:
                    np.save(fnp, np.reshape(xtra_by_seq[src_seq][tar_seq], (-1, nb_mel)) * mvn_params["std"] +
                            mvn_params["mean"])

            plot_x([xtra_by_seq[src_seq][seq] for seq in seqs], seq_names,
                   "%s/test/img/x_tra/%s_tra.png" % (expdir, src_seq_name), True)

    if True:
        # tsne z1 and z2
        print("t-SNE analysis on latent variables")
        n = [len(z1_by_seq[seq]) for seq in seqs]
        z1 = np.concatenate([z1_by_seq[seq] for seq in seqs], axis=0)
        z2 = np.concatenate([z2_by_seq[seq] for seq in seqs], axis=0)

        p = 30
        print("  perplexity = %s" % p)
        tsne = TSNE(n_components=2, verbose=0, perplexity=p, n_iter=1000)
        z1_tsne = _unflatten(tsne.fit_transform(z1), n)
        scatter_plot(z1_tsne, seq_names, "z1_tsne_%03d" % p,
                     "%s/test/img/z1_tsne_%03d.png" % (expdir, p))
        z2_tsne = _unflatten(tsne.fit_transform(z2), n)
        scatter_plot(z2_tsne, seq_names, "z2_tsne_%03d" % p,
                     "%s/test/img/z2_tsne_%03d.png" % (expdir, p))

def tsne_by_label(expdir, model, conf, iterator_by_seqs, seqs, dt_dset):

    if len(seqs) > 25:
        seqs = sorted(list(np.random.choice(seqs, 25, replace=False)))

    # infer z1, z2
    z1_by_seq = defaultdict(list)
    z2_by_seq = defaultdict(list)
    for seq in seqs:
        for x, _, _, _, _ in iterator_by_seqs([seq], bs=conf['batch_size']):
            x = tf.stack(tf.cast(x, dtype=tf.float32), axis=0)
            _, _, _, _, _, _, qz1_x, qz2_x = model.encoder(x)
            z2_by_seq[seq].append(qz2_x[0])
            z1_by_seq[seq].append(qz1_x[0])

        z1_by_seq[seq] = np.concatenate(z1_by_seq[seq], axis=0)
        z2_by_seq[seq] = np.concatenate(z2_by_seq[seq], axis=0)

    # tsne z1 and z2
    print("t-SNE analysis on latent variables by label")
    n = [len(z1_by_seq[seq]) for seq in seqs]
    z1 = np.concatenate([z1_by_seq[seq] for seq in seqs], axis=0)
    z2 = np.concatenate([z2_by_seq[seq] for seq in seqs], axis=0)

    p = 30
    tsne = TSNE(n_components=2, verbose=0, perplexity=p, n_iter=1000)
    z1_tsne_by_seq = dict(list(zip(seqs, _unflatten(tsne.fit_transform(z1), n))))

    for gen_fac, seq2lab in list(dt_dset.labs_d.items()):
        _labs, _z1 = _join(z1_tsne_by_seq, seq2lab)
        scatter_plot(_z1, _labs, gen_fac,
                     "%s/test/img/tsne_by_label_z1_%s_%03d.png" % (expdir, gen_fac, p))

    z2_tsne_by_seq = dict(list(zip(seqs, _unflatten(tsne.fit_transform(z2), n))))
    for gen_fac, seq2lab in list(dt_dset.labs_d.items()):
        _labs, _z2 = _join(z2_tsne_by_seq, seq2lab)
        scatter_plot(_z2, _labs, gen_fac,
                     "%s/test/img/tsne_by_label_z2_%s_%03d.png" % (expdir, gen_fac, p))

    for gen_talab, seq2talabseq in list(dt_dset.talabseqs_d.items()):
        _talabs, _z1 = _join_talab(z1_tsne_by_seq, seq2talabseq.seq2talabseq, dt_dset.talab_vals[gen_talab])
        scatter_plot(_z1, _talabs, gen_talab,
                     "%s/test/img/tsne_by_label_z1_%s_%03d.png" % (expdir, gen_talab, p))

    for gen_talab, seq2talabseq in list(dt_dset.talabseqs_d.items()):
        _talabs, _z2 = _join_talab(z2_tsne_by_seq, seq2talabseq.seq2talabseq, dt_dset.talab_vals[gen_talab])
        scatter_plot(_z2, _talabs, gen_talab,
                     "%s/test/img/tsne_by_label_z2_%s_%03d.png" % (expdir, gen_talab, p))


def estimate_mu2_dict(model, conf, iterator):
    """
    estimate mu2 for sequences produced by iterator
    Args: model(FHVAE):
          iterator(Callable):
    Return: mu2_dict(dict): sequence index to mu2 dict
    """

    nseg_table = defaultdict(float)
    z2_sum_table = defaultdict(float)

    for x_val, y_val, _, _, _ in iterator(bs=conf['batch_size']):
        z2_mu = model.encoder.z2_mu_separate(tf.stack(tf.cast(x_val, dtype=tf.float32), axis=0))

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


def _print_mu2_stat(mu2_dict):
    norm_sum = 0.
    dim_norm_sum = 0.
    for y in sorted(mu2_dict.keys()):
        norm_sum += np.linalg.norm(mu2_dict[y])
        dim_norm_sum += np.abs(mu2_dict[y])
    avg_norm = norm_sum / len(mu2_dict)
    avg_dim_norm = dim_norm_sum / len(mu2_dict)
    print("avg. norm = %.2f, #mu2 = %s" % (avg_norm, len(mu2_dict)))
    print("per dim: %s" % (" ".join(["%.2f" % v for v in avg_dim_norm]),))


def _softmax(x):
    ## First column are zeros (as added in fix_logits in model, so leave these out and return size-1 tens
    #y = np.exp(x[:, 1:])
    #return y / np.sum(y, axis=1, keepdims=True)
    return tf.nn.softmax(x, axis=1)


def _seq_translate(model, tr_shape, src_z1, src_z2, del_mu2):
    mod_z2 = src_z2 + del_mu2[np.newaxis, ...]
    _, _, _, _, px_z = model.decoder(\
        np.zeros((len(src_z1), tr_shape[0], tr_shape[1]), dtype=np.float32), src_z1, mod_z2)

    return px_z[0]


def _unflatten(l_flat, n_l):
    """
    unflatten a list
    """
    l = []
    offset = 0
    for n in n_l:
        l.append(l_flat[offset:offset+n])
        offset += n
    assert(offset == len(l_flat))
    return l


def _join(z_by_seqs, seq2lab):
    d = defaultdict(list)
    for seq, z in list(z_by_seqs.items()):
        d[seq2lab[seq]].append(z)
    for lab in d:
        d[lab] = np.concatenate(d[lab], axis=0)
    return list(d.keys()), list(d.values())


def _join_talab(z_by_seqs, seq2talabseq, talab_vals):
    d = defaultdict(list)
    for seq, z in list(z_by_seqs.items()):
        n_segs = z.shape[0]
        seq_talabs = seq2talabseq[seq].talabs
        for seg in range(n_segs):
            idx = seq_talabs[seg].lab
            talab = list(talab_vals.keys())[list(talab_vals.values()).index(idx)]
            d[talab].append(z[seg, :])
    for lab in d:
        d[lab] = np.stack(d[lab], axis=0)
    return list(d.keys()), list(d.values())
