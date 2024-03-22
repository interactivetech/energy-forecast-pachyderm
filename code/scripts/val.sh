#!/bin/bash

# Paths to the model and data
MODEL_PATH="/pfs/train/TCN_model.pt"
TRAIN_PKL_PATH="/pfs/process/train_en_transformed2.pkl"
VAL_PKL_PATH="/pfs/process/val_en_transformed2.pkl"
DAY_SERIES_PKL_PATH="/pfs/process/day_series2.pkl"
FIGURE_PATH="/pfs/out/validation_figure.png"

# Execute the validation script
python /pfs/code/code/py/val.py \
  --model_path "$MODEL_PATH" \
  --train_pkl "$TRAIN_PKL_PATH" \
  --val_pkl "$VAL_PKL_PATH" \
  --day_series_pkl "$DAY_SERIES_PKL_PATH" \
  --figure_path "$FIGURE_PATH"