import argparse
import pandas as pd
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import pickle
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings
warnings.filterwarnings("ignore")

def process_energy_data(input_csv, train_pkl, val_pkl, day_series_pkl,series_en_pkl):
    logging.info("Starting to process the dataset.")

    # Load the dataset
    df = pd.read_csv(input_csv, parse_dates=['time'])
    df.set_index('time', inplace=True)

    # Process the dataset
    df_day_avg = df.resample('D').mean()
    series_en = TimeSeries.from_dataframe(df_day_avg, value_cols=['generation hydro run-of-river and poundage']) 
    
    # Split the data
    train_en, val_en = series_en.split_after(pd.Timestamp("20170901"))
    
    # Scale the data
    scaler_en = Scaler()
    train_en_transformed = scaler_en.fit_transform(train_en)
    val_en_transformed = scaler_en.transform(val_en)
    
    # Day series as covariate
    day_series = datetime_attribute_timeseries(series_en, attribute="day", one_hot=True)
    
    # Save the transformed series
    with open(train_pkl, 'wb') as f:
        pickle.dump(train_en_transformed, f)
        logging.info(f"Saved transformed training data to {train_pkl}.")
    with open(val_pkl, 'wb') as f:
        pickle.dump(val_en_transformed, f)
        logging.info(f"Saved transformed validation data to {val_pkl}.")
    with open(day_series_pkl, 'wb') as f:
        pickle.dump(day_series, f)
        logging.info(f"Saved day series to {day_series_pkl}.")
    with open(series_en_pkl, 'wb') as f:
        pickle.dump(series_en, f)
        logging.info(f"Saved day series to {series_en_pkl}.")

    logging.info("Finished processing the dataset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process energy dataset and save transformed data.")
    parser.add_argument("--input_csv", type=str, help="Path to input CSV file.")
    parser.add_argument("--train_pkl", type=str, help="Path to save train_en_transformed.pkl.")
    parser.add_argument("--val_pkl", type=str, help="Path to save val_en_transformed.pkl.")
    parser.add_argument("--day_series_pkl", type=str, help="Path to save day_series.pkl.")
    parser.add_argument("--series_en_transformed_pkl", type=str, help="Path to save series_en_transformed.pkl.")
    
    args = parser.parse_args()
    
    process_energy_data(args.input_csv, args.train_pkl, args.val_pkl, args.day_series_pkl,args.series_en_transformed_pkl)