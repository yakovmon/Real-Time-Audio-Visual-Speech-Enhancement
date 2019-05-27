#!/bin/bash
# Activate virtual environment
source /cs/engproj/322/new_avse_venv/bin/activate
# Activate modules
source /cs/engproj/322/load_module.sh
# Preprocess train data
python /cs/engproj/322/avse/speech_enhancer.py -bd /cs/engproj/322/out_check preprocess -dn check1 -ds /cs/engproj/322/data -n /cs/engproj/322/raw_noise/libri-speech/train -s check1
