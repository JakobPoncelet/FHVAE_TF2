# FHVAE_TF2
## Implementation
Implementation of Scalable Factorised Hierarchical Variational Autoencoder (https://github.com/wnhsu/ScalableFHVAE) in Tensorflow 2.0 and Python 3.6.8. It is not verified yet if the code is 100% correct compared to the original code!!

## Model
It is possible to use the LSTM model of the paper, or use a Transformer for the encoder (still quite experimental, results not convincing yet). 
Regularizations have been added to Z1 and to Z2, with a cross-entropy loss based on classification into labels. For now, sequence factors like gender, region and speaker recognition are added on Z2, and segmentspecific time-aligned labels ('talabs') like phones and phoneclass recognition are added on Z1. Wav-files without labels are also allowed (for training with unsupervised data).

## Supported datasets
The TIMIT database can be fully replicated using the example script. 
CGN also has a preparation script (of which some part has to run in Matlab).

## How to run
It is advised to start with TIMIT and go through the code step by step.
1) Preprocessing examples --> prepare_timit  (this will set up your database and create all necessary files)
2) Change the template config.cfg file to your needs or use one in the configs-directory.
3) Scripts/train --> python run_hs_train.py --expdir=... --config=...
4) Scripts/test --> python run_eval.py --expdir=...

## Running training on Condor
The expdir has to exist already: 
     condor_submit jobfile.job expdir=... config=... 
(For TIMIT: can take 2 days)

## Contact
jakob.poncelet[at]esat.kuleuven.be

## References
Hsu, W. N., Zhang, Y., and Glass, J. Unsupervised learning of disentangled and interpretable representations from sequential data. In NIPS, 2017.

Hsu, W. N. and Glass, J. Scalable factorized hierarchical variational autoencoder training. In Interspeech, 2018.
