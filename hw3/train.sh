#!/bin/bash

#THEANO_FLAGS=device=gpu,floatX=float32 python3 CNN.py
python3 CNN.py $1 $2
python3 self_training.py $1 $2



