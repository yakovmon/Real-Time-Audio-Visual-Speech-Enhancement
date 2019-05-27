#!/bin/bash
# Activate virtual environment
source /cs/engproj/322/avse_virtualenv/bin/activate
# Activate Tensorflow
# module load tensorflow
# Preprocess the data
python /cs/engproj/322/avse/speech_enhancer.py -bd /cs/engproj/322/test_network/out_yakov_small_train preprocess -dn yakov_small_test -ds /cs/engproj/322/data -n /cs/engproj/322/raw_noise/libri-speech/train -s yakov_small_test
