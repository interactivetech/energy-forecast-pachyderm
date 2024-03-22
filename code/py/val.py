import argparse
import logging
import matplotlib.pyplot as plt
import pickle
from darts.models import TCNModel
from darts.metrics import rmse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_model(model_path, train_pkl, val_pkl, day_series_pkl, figure_path):
    logging.info("Loading data and model for validation...")
    with open(train_pkl, 'rb') as f:
        train_en_transformed = pickle.load(f)
    with open(val_pkl, 'rb') as f:
        val_en_transformed = pickle.load(f)
    with open(day_series_pkl, 'rb') as f:
        day_series = pickle.load(f)
    model_en = TCNModel.load(model_path)

    logging.info("Performing validation...")
    # Perform forecasts
    backtest_train = model_en.historical_forecasts(
        series=train_en_transformed,
        past_covariates=day_series,
        forecast_horizon=7,
        retrain=False,
        verbose=True,
        overlap_end=False
    )
    backtest_val = model_en.historical_forecasts(
        series=val_en_transformed,
        past_covariates=day_series,
        forecast_horizon=7,
        retrain=False,
        verbose=True,
        overlap_end=False
    )

    # Compute metrics
    rmse_train = round(rmse(train_en_transformed, backtest_train), 2)
    rmse_val = round(rmse(val_en_transformed, backtest_val), 2)
    logging.info(f"Validation finished. RMSE on Training Set: {rmse_train}, RMSE on Validation Set: {rmse_val}")

    # Plot results
    plt.figure(figsize=(15, 5))

    # Training set plot
    plt.subplot(1, 2, 1)
    train_en_transformed.plot(label="Actual - Train")
    backtest_train.plot(label="Forecast - Train")
    plt.title('Training Set Forecasts '+f'RMSE: {rmse_train}')
    plt.legend()

    # Validation set plot
    plt.subplot(1, 2, 2)
    val_en_transformed.plot(label="Actual - Validation")
    backtest_val.plot(label="Forecast - Validation")
    plt.title('Validation Set Forecasts '+f'RMSE: {rmse_val}')
    plt.legend()

    plt.savefig(figure_path)
    logging.info(f"Figures saved to {figure_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate TCN model on energy dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained TCN model.")
    parser.add_argument("--train_pkl", type=str, required=True, help="Path to train_en_transformed.pkl.")
    parser.add_argument("--val_pkl", type=str, required=True, help="Path to val_en_transformed.pkl.")
    parser.add_argument("--day_series_pkl", type=str, required=True, help="Path to day_series.pkl.")
    parser.add_argument("--figure_path", type=str, required=True, help="Path to save the matplotlib figures.")
    args = parser.parse_args()

    validate_model(args.model_path, args.train_pkl, args.val_pkl, args.day_series_pkl, args.figure_path)
