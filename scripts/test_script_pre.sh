#!/bin/bash
# Activate virtual environment
source /cs/engproj/322/new_avse_venv/bin/activate
# Activate modules
source /cs/engproj/322/scripts/load_module.sh
# Preprocess test data
python /cs/engproj/322/avse/speech_enhancer.py -bd /cs/engproj/322/out_preprocess_english preprocess -dn yakov_final_test -ds /cs/engproj/322/data -n /cs/engproj/322/raw_noise/english_noise -s yakov_final_test
