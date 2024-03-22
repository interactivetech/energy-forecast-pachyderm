#!/bin/bash

# Paths to the data and model
PRETRAINED_MODEL_PATH="/pfs/model/TCN_model.pt"
SAVED_MODEL_PATH="/pfs/out/TCN_model.pt"
FIGURE_PATH="/pfs/out/training_figure.png"
TRAIN_PKL_PATH="/pfs/process/train_en_transformed2.pkl"
VAL_PKL_PATH="/pfs/process/val_en_transformed2.pkl"
DAY_SERIES_PKL_PATH="/pfs/process/day_series2.pkl"
SERIES_EN_PKL_PATH="/pfs/process/series_en.pkl"
# Execute the training script
python /pfs/code/code/py/train.py \
  --pretrained_path "$PRETRAINED_MODEL_PATH" \
  --save_model_path "$SAVED_MODEL_PATH" \
  --figure_path "$FIGURE_PATH" \
  --train_pkl_path "$TRAIN_PKL_PATH" \
  --val_pkl_path "$VAL_PKL_PATH" \
  --day_series_pkl_path "$DAY_SERIES_PKL_PATH" \
  --series_en_pkl_path "$SERIES_EN_PKL_PATH"