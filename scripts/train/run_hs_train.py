from __future__ import absolute_import
import os
import sys
print(sys.path)
print(os.getcwd())
sys.path.insert(0, os.getcwd())
import time
import argparse
import tensorflow as tf
import numpy as np
import pickle
import ast
import shutil
from collections import OrderedDict
from configparser import ConfigParser
from scripts.train.hs_train_loaders import load_data_reg
from fhvae.runners.hs_train_fhvae import hs_train_reg
from fhvae.models.reg_fhvae_lstm import RegFHVAEnew
from fhvae.models.reg_fhvae_transf import RegFHVAEtransf

# For debugging on different GPU: os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

'''
Commands (pycharm setup)
Script path:
/users/spraak/jponcele/JakobFHVAE/scripts/train/run_hs_train.py

Parameters:
--expdir /esat/spchdisk/scratch/jponcele/fhvae_jakob/exp --config /users/spraak/jponcele/JakobFHVAE/configs/timit/config_lstm_complete.cfg
Working directory:
/users/spraak/jponcele/JakobFHVAE
'''

def main(expdir, configfile):
    ''' main function '''

    # read and copy the config file, change location if necessary
    if os.path.exists(os.path.join(expdir, 'config.cfg')):
        print("Expdir already contains a config file... Overwriting!")
        os.remove(os.path.join(expdir, 'config.cfg'))

    shutil.copyfile(configfile, os.path.join(expdir, 'config.cfg'))
    conf = load_config(os.path.join(expdir, 'config.cfg'))
    conf['expdir'] = expdir

    # symbolic link dataset to ./datasets
    if os.path.islink(os.path.join("./datasets", conf['dataset'])):
        os.unlink(os.path.join("./datasets", conf['dataset']))
    if os.path.isdir(os.path.join("./datasets", conf['dataset'])):
        print("Physical directory already exists in ./datasets, cannot create symlink of same name to the dataset.")
        exit(1)
    os.symlink(os.path.join(conf['datadir'], conf['dataset']), os.path.join("./datasets", conf['dataset']))

    # set up the iterators over the dataset (for large datasets this may take a while)
    tr_nseqs, tr_shape, sample_tr_seqs, tr_iterator_by_seqs, dt_iterator, tr_dset = \
        load_data_reg(conf['dataset'], conf['fac_root'], conf['facs'], conf['talabs'])

    # identify regularizing factors
    used_labs = conf['facs'].split(':')
    lab2idx = {name:tr_dset.labs_d[name].lablist for name in used_labs}
    print("labels and indices of facs: ", lab2idx)
    conf['lab2idx'] = lab2idx

    used_talabs = conf['talabs'].split(':')
    conf['talab_vals'] = tr_dset.talab_vals
    print("labels and indices of talabs: ", tr_dset.talab_vals)

    # For now, apply talabs on z1 (e.g. time-aligned phones) and labs on z2 (e.g. region, gender).
    # To switch, swap b_n/c_n in z_nlabs and b/c in _make_batch of hs_train_loaders
    # (will also require changes to eval script)
    c_n = OrderedDict([(lab, tr_dset.labs_d[lab].nclass) for lab in used_labs])
    b_n = OrderedDict([(talab, tr_dset.talabseqs_d[talab].nclass) for talab in used_talabs])

    # save input shape [e.g. tuple (20,80)] and numclasses for testing phase
    conf['tr_shape'] = tr_shape
    conf['c_n'] = c_n
    conf['b_n'] = b_n

    # whether to train hierarchically on a random subset of nmu2 sequences in every epoch, or on all data every epoch
    if conf['training'] != 'hierarchical':  # i.e. training = 'normal'
        conf['nmu2'] = len(tr_dset.seqlist)

    # dump settings
    with open(os.path.join(expdir, 'trainconf.pkl'), "wb") as fid:
        pickle.dump(conf, fid)

    # initialize the model
    if conf['model'] == 'LSTM':
        model = RegFHVAEnew(z1_dim=conf['z1_dim'], z2_dim=conf['z2_dim'], z1_rhus=conf['z1_rhus'], z2_rhus=conf['z2_rhus'], \
                            x_rhus=conf['x_rhus'], nmu2=conf['nmu2'], z1_nlabs=b_n, z2_nlabs=c_n, \
                            mu_nl=None, logvar_nl=None, tr_shape=tr_shape, bs=conf['batch_size'], \
                            alpha_dis=conf['alpha_dis'], alpha_reg_b=conf['alpha_reg_b'], alpha_reg_c=conf['alpha_reg_c'])
    if conf['model'] == 'transformer':
        model = RegFHVAEtransf(z1_dim=conf['z1_dim'], z2_dim=conf['z2_dim'], nmu2=conf['nmu2'], x_rhus=conf['x_rhus'], \
                            tr_shape=tr_shape, z1_nlabs=b_n, z2_nlabs=c_n, mu_nl=None, logvar_nl=None, \
                            d_model=conf['d_model'], num_enc_layers=conf['num_enc_layers'], num_heads=conf['num_heads'], \
                            dff=conf['dff'], pe_max_len=conf['pe_max_len'], rate=conf['rate'], \
                            alpha_dis=conf['alpha_dis'], alpha_reg_b=conf['alpha_reg_b'], alpha_reg_c=conf['alpha_reg_c'])

    # START
    hs_train_reg(expdir, model, conf, sample_tr_seqs, tr_iterator_by_seqs, dt_iterator, tr_dset)


def load_config(conf):
    ''' Load configfile and extract arguments as a dict '''
    cfgfile = ConfigParser(interpolation=None)
    cfgfile.read(conf)
    train_conf = dict(cfgfile.items('RegFHVAE'))
    for key, val in train_conf.items():
        try:
            # get numbers as int/float/list
            train_conf[key] = ast.literal_eval(val)
        except:
            # text / paths
            pass
    return train_conf


if __name__ == '__main__':

    print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

    #parse the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--expdir", type=str, default="./exp",
                        help="where to store the experiment")
    parser.add_argument("--config", type=str, default="./config.cfg",
                        help="config file for the experiment")
    args = parser.parse_args()

    if os.path.isdir(args.expdir):
        print("Expdir already exists")
    os.makedirs(args.expdir, exist_ok=True)

    main(args.expdir, args.config)
