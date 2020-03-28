# FHVAE_TF2
Implementation of Scalable Factorised Hierarchical Variational Autoencoder (https://github.com/wnhsu/ScalableFHVAE) in Tensorflow 2.0 and Python 3.6.8.   
**Important notice:** It is not verified yet if the code is 100% correct compared to the original code!!

## Model
It is possible to use the LSTM model of the paper, or use a Transformer for the encoder (still quite experimental, results not convincing yet). 
Regularizations have been added to Z1 and to Z2, with a cross-entropy loss based on classification into labels. For now, sequence factors like gender, region and speaker recognition are added on Z2, and segmentspecific time-aligned labels ('talabs') like phones and phoneclass recognition are added on Z1. Wav-files without labels are also allowed (for training with unsupervised data).

## Supported datasets
The TIMIT database can be fully replicated using the example script. 
CGN also has a preparation script (of which some part has to run in Matlab).

## Installation

### Requirements
Please make sure you are running Python 3.6.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Getting started

It is advised to start with TIMIT and go through the code step by step.

#### 1. Preprocessing
```
python preprocessing_examples/prepare_timit_numpy.py <data> <doc> --ftype ... --out_dir ...
```
This will set up your database and create all files needed for training.

#### 2. Training
Make sure `expdir` exists. For the hyperparameter setup, you can modify the template `config.cfg` file or use one in the configs-directory.
```bash
python scripts/train/run_hs_train.py --expdir=... --config=...
```

#### 3. Evaluation
```bash
python scripts/test/run_eval.py --expdir=...
```

## Running training on Condor
The expdir has to exist already:   
```bash
condor_submit jobfile.job expdir=... config=...   
```
For TIMIT training can take up to 2 days.  
CGN training can take up to 4 days, depending on the components chosen.

## Contact
jakob.poncelet@esat.kuleuven.be

## References
Hsu, W. N., Zhang, Y., and Glass, J. Unsupervised learning of disentangled and interpretable representations from sequential data. In NIPS, 2017.

Hsu, W. N. and  Glass, J. Scalable  factorized  hierarchical  variational autoencoder training. In Interspeech, 2018.
