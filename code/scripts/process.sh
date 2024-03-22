#!/bin/bash

# Variables for paths, change these according to your file locations
INPUT_CSV_PATH="/pfs/data/energy_dataset.csv"
TRAIN_PKL_PATH="/pfs/out/train_en_transformed2.pkl"
VAL_PKL_PATH="/pfs/out/val_en_transformed2.pkl"
DAY_SERIES_PKL_PATH="/pfs/out/day_series2.pkl"
SERIES_EN_PKL_PATH="/pfs/out/series_en.pkl"

# Running the Python script
python /pfs/code/code/py/process.py --input_csv "$INPUT_CSV_PATH" \
  --train_pkl "$TRAIN_PKL_PATH" \
  --val_pkl "$VAL_PKL_PATH" \
  --day_series_pkl "$DAY_SERIES_PKL_PATH" \
  --series_en_transformed_pkl "$SERIES_EN_PKL_PATH"