#!/bin/bash
# Activate virtual environment
source /cs/engproj/322/new_avse_venv/bin/activate
# Activate modules
source /cs/engproj/322/load_module.sh
# Train the model
python /cs/engproj/322/avse/speech_enhancer.py -bd /cs/engproj/322/out_train_model train -mn yakov_train_model -tdn /cs/engproj/322/out_preprocess_train/cache/preprocessed/yakov_train -vdn /cs/engproj/322/out_preprocess_validation/cache/preprocessed/yakov_validation
