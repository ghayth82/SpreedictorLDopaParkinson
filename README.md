# ParkinsonDream
## Requirements
To run the scripts you need the following software requirements:
1. Install [Anaconda2-4.4.0](https://www.continuum.io/downloads). 

The requirements that were used to train and evaluate the models can
be loaded via
```
conda create --name <env> --file requirements.txt
```

## Environment variables
You need to set the environment variable `PARKINSON_DREAM_LDOPA_DATA` to point to
the directory where the dataset should be stored. For instance,
on Linux use 

`export PARKINSON_DREAM_LDOPA_DATA=/path/to/data/`

Furthermore, you might need to set `KERAS_BACKEND` to utilize `tensorflow`
rather than e.g. `theano` according to 
`export KERAS_BACKEND=tensorflow`

or by running python with

`KERAS_BACKEND=tensorflow python <script.py>`

## Training

To train the models we invoked
```
cd <repo_root>/code

# Subchallenge 2.1: tremorScore
# variant 1
python run_all.py -df fh_0.5-tre-all -mf metatime_deep_conv_v2 --allaug
# variant 2
python run_all.py -df raw-tre-all -mf metatime_deep_conv_v2 --allaug

# Subchallenge 2.2: dyskinesia
# variant 1
python run_all.py -df raw-dys-all -mf metatime_conv2l_70_200_10_50_30_20_10 --allaug
# variant 2
python run_all.py -df raw-dys-all -mf metatime_deep_conv_v2 --allaug

# Subchallenge 2.3: bradykinesia
# variant 1
python run_all.py -df raw-bra-all -mf metatime_deep_conv_v2 --allaug_v2
# variant 2
python run_all.py -df fh_0.5-bra-all -mf metatime_deep_conv_v2 --allaug

```
These commands will automatically download and preprocess the LDopa dataset
provided for the subchallenges 2.1-2.3.

## Feature prediction

Finally, the feature predictions were generated and submitted to the challenge
submission queue with
```
python featurizer.py tre1  --gen --submit
python featurizer.py tre2  --gen --submit
python featurizer.py bra1  --gen --submit
python featurizer.py bra2  --gen --submit
python featurizer.py dys1  --gen --submit
python featurizer.py dys2  --gen --submit
```
