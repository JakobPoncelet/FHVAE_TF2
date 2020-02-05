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
from fhvae.runners.hs_train_fhvae_tf2 import hs_train_reg
from fhvae.models.reg_fhvae_tf2 import RegFHVAEnew

# For debugging on different GPU: os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

'''
Commands
Script path:
/users/spraak/jponcele/JakobFHVAE/scripts/train/run_hs_train.py

Parameters:
--expdir /esat/spchdisk/scratch/jponcele/fhvae_jakob/exp
Working directory:
/users/spraak/jponcele/JakobFHVAE
'''

def main(expdir):
    ''' main function '''

    # read and copy the config file, change location if necessary
    shutil.copyfile('./config.cfg', os.path.join(expdir, 'config.cfg'))
    conf = load_config(os.path.join(expdir, 'config.cfg'))
    conf['expdir'] = expdir

    # symbolic link dataset to ./datasets
    if os.path.islink(os.path.join("./datasets", conf['dataset'])):
        os.unlink(os.path.join("./datasets", conf['dataset']))
    if os.path.isdir(os.path.join("./datasets", conf['dataset'])):
        print("Physical directory already exists in ./datasets, cannot create symlink of same name to the dataset.")
        exit(1)
    os.symlink(os.path.join(conf['datadir'], conf['dataset']), os.path.join("./datasets", conf['dataset']))

    # set up the iterators over the dataset
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

    c_n = OrderedDict([(lab, tr_dset.labs_d[lab].nclass) for lab in used_labs])

    # FOR NOW USE THE LABELS FOR z2 ALSO FOR z1 TO TEST IF IT WORKS (with alpha_reg_b = 0 to prevent regularizing z1)
    #b_n = c_n
    b_n = OrderedDict([(talab, tr_dset.talabseqs_d[talab].nclass) for talab in used_talabs])

    # save input shape [e.g. tuple (20,80)] and numclasses for testing phase
    conf['tr_shape'] = tr_shape
    conf['c_n'] = c_n
    conf['b_n'] = b_n

    with open(os.path.join(expdir, 'trainconf.pkl'), "wb") as fid:
        pickle.dump(conf, fid)

    # initialize the model
    model = RegFHVAEnew(z1_dim=conf['z1_dim'], z2_dim=conf['z2_dim'], z1_rhus=conf['z1_rhus'], z2_rhus=conf['z2_rhus'], \
                        x_rhus=conf['x_rhus'], nmu2=conf['nmu2'], z1_nlabs=b_n, z2_nlabs=c_n, \
                        mu_nl=None, logvar_nl=None, tr_shape=tr_shape, alpha_dis=conf['alpha_dis'], \
                        alpha_reg_b=conf['alpha_reg_b'], alpha_reg_c=conf['alpha_reg_c'])


    # START
    hs_train_reg(expdir, model, conf, sample_tr_seqs, tr_iterator_by_seqs, dt_iterator)


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
    args = parser.parse_args()

    if os.path.isdir(args.expdir):
        print("Expdir already exists")
    os.makedirs(args.expdir, exist_ok=True)

    main(args.expdir)
