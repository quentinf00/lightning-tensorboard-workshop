# Pytorch Lightning & Tensorboard workshop

## Prerequisites
 - python 3
 - virtualenv

## Install

```
virtualenv venv python=python3
source venv/bin/activate
pip install -r requirements.txt
```

## Workshop points
I/ Intro Démo

1: Code walkthrough
2: Training
3: logs
4: Testing logs

II/Diagnostic
- log metric and hparams
- display confusion matrix
- display wrongly classified images

III/Technical logging
- Profiler
- log model graph
- log lr
- log gradients


From there:

same training we will:

- add checkpointing
- gradient accumulation
- tune lr
- logging
- multigpu training
- mutli node training
- tpu training
- mixed precision training
- gradient clipping

debugging utils:
- fast\_dev\_run
- limit batches
- track grad norm
- log gpu memory

## Pytorch training 

### Ideas
- First training
