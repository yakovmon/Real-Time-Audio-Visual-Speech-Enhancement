#!/bin/bash
# Activate virtual environment
source /cs/engproj/322/new_avse_venv/bin/activate
# Activate modules
source /cs/engproj/322/scripts/load_module.sh
# Predict
python /cs/engproj/322/avse/speech_enhancer.py -bd /cs/engproj/322/out_predict_final_test predict -mn /cs/engproj/322/out_train_model/cache/models/yakov_train_model -dn /cs/engproj/322/out_preprocess_english/cache/preprocessed/yakov_final_test
