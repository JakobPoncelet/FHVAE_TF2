import sys
import numpy as np
from fhvae.datasets.seg_dataset import NumpySegmentDataset


def load_data_reg(name, seqlist_path=None, lab_names=None, talab_names=None):
    # lab_names e.g. region:gender then loaded from (seqlist_path % lab_name) as scp file
    root = "./datasets/%s" % name
    mvn_path = "%s/train/mvn.pkl" % root
    seg_len = 20  # 15
    seg_shift = 8  # 5

    Dataset = NumpySegmentDataset

    if lab_names is not None:
        lab_specs = [(lab, seqlist_path % lab) for lab in lab_names.split(':')]
    else:
        lab_specs = list()
    if talab_names is not None:
        talab_specs = [(talab, seqlist_path % talab) for talab in talab_names.split(':')]
    else:
        talab_specs = list()

    # initialize the datasets
    tr_dset = Dataset(
        "%s/train/feats.scp" % root, "%s/train/len.scp" % root,
        lab_specs=lab_specs, talab_specs=talab_specs,
        min_len=seg_len, preload=False, mvn_path=mvn_path,
        seg_len=seg_len, seg_shift=seg_shift, rand_seg=True)

    dt_dset = Dataset(
        "%s/dev/feats.scp" % root, "%s/dev/len.scp" % root,
        lab_specs=lab_specs, talab_specs=talab_specs,
        min_len=seg_len, preload=False, mvn_path=mvn_path,
        seg_len=seg_len, seg_shift=seg_len, rand_seg=False,
        copy_from=tr_dset)

    return _load_reg(tr_dset, dt_dset) + (tr_dset,)


def _load_reg(tr_dset, dt_dset):
    def _make_batch(seqs, feats, nsegs, seq2idx, seq2reg, talabs):
        # what the iterator returns --> xval, yval, nval, cval
        x = feats
        y = np.asarray([seq2idx[seq] for seq in seqs])
        n = np.asarray(nsegs)
        c = np.asarray([seq2reg[seq] for seq in seqs])
        b = np.asarray(talabs)
        return x, y, n, c, b

    def sample_tr_seqs(nseqs):
        # randomly sample nseqs amount of sequences from the train set
        return np.random.choice(tr_dset.seqlist, nseqs, replace=False)

    tr_nseqs = len(tr_dset.seqlist)
    tr_shape = tr_dset.get_shape()

    def tr_iterator_by_seqs(s_seqs, bs=256, seg_rem=False):
        # build an iterator over the dataset, into batches
        seq2idx = dict([(seq, i) for i, seq in enumerate(s_seqs)])
        lab_names = list(tr_dset.labs_d.keys())
        talab_names = list(tr_dset.talabseqs_d.keys())
        ii = list()
        for k in s_seqs:
            itm = [tr_dset.labs_d[name].lablist.index(tr_dset.labs_d[name].seq2lab[k]) for name in lab_names]
            ii.append(np.asarray(itm))
        seq2regidx = dict(list(zip(s_seqs, ii)))
        _iterator = tr_dset.iterator(bs, seg_shuffle=True, seg_rem=seg_rem, seqs=s_seqs, lab_names=lab_names, talab_names=talab_names)
        for seqs, feats, nsegs, labs, talabs in _iterator:
            yield _make_batch(seqs, feats, nsegs, seq2idx, seq2regidx, talabs)

    def dt_iterator(bs=256):
        # development set iterator for validation step
        seq2idx = dict([(seq, i) for i, seq in enumerate(dt_dset.seqlist)])
        lab_names = list(dt_dset.labs_d.keys())
        talab_names = list(dt_dset.talabseqs_d.keys())
        ii = list()
        for k in dt_dset.seqlist:
            itm = [dt_dset.labs_d[name].lablist.index(dt_dset.labs_d[name].seq2lab[k]) for name in lab_names]
            ii.append(np.asarray(itm))
        seq2regidx = dict(list(zip(dt_dset.seqlist, ii)))
        _iterator = dt_dset.iterator(bs, seg_shuffle=False, seg_rem=True, seqs=dt_dset.seqlist, lab_names=lab_names, talab_names=talab_names)
        for seqs, feats, nsegs, labs, talabs in _iterator:
            yield _make_batch(seqs, feats, nsegs, seq2idx, seq2regidx, talabs)

    return tr_nseqs, tr_shape, sample_tr_seqs, tr_iterator_by_seqs, dt_iterator
