import argparse
import logging
from darts.models import TCNModel
import matplotlib.pyplot as plt
import pickle
from darts import TimeSeries, concatenate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_torch_kwargs():
    # Placeholder for torch kwargs if needed
    return {}

def train_model(pretrained_path, save_model_path, figure_path, train_pkl_path, val_pkl_path, day_series_pkl_path,series_en_pkl_path):
    logging.info("Loading data...")
    # Load the preprocessed data
    with open(train_pkl_path, 'rb') as f:
        train_en_transformed = pickle.load(f)
    with open(val_pkl_path, 'rb') as f:
        val_en_transformed = pickle.load(f)
    with open(day_series_pkl_path, 'rb') as f:
        day_series = pickle.load(f)
    # series_en_pkl_path
    with open(series_en_pkl_path, 'rb') as f:
        series_en = pickle.load(f)
    logging.info("Data loaded successfully.")

    # Load the pretrained model
    logging.info("Loading pretrained model...")
    model_name = "TCN_energy"
    model_en = TCNModel(
        input_chunk_length=365,
        output_chunk_length=7,
        n_epochs=50,
        dropout=0.2,
        dilation_base=2,
        weight_norm=True,
        kernel_size=5,
        num_filters=8,
        nr_epochs_val_period=1,
        random_state=0,
        save_checkpoints=True,
        model_name=model_name,
        force_reset=True,
    **generate_torch_kwargs()
    )
    model_en.load_weights(pretrained_path)
    model_en.n_epochs = 2

    # Start training
    logging.info("Starting training...")
    model_en.fit(
        series=train_en_transformed,
        past_covariates=day_series,
        val_series=val_en_transformed,
        val_past_covariates=day_series,
    )
    logging.info("Training completed.")
    
    # Save the trained model
    model_en.save(save_model_path)
    logging.info(f"Model saved to {save_model_path}.")

    # Perform and plot backtest
    # series_en_transformed = train_en_transformed + val_en_transformed
    backtest_en = model_en.historical_forecasts(
        series=series_en,
        past_covariates=day_series,
        start=val_en_transformed.start_time(),
        forecast_horizon=7,
        stride=7,
        last_points_only=False,
        retrain=False,
        verbose=True,
    )
    backtest_en = concatenate(backtest_en)
    plt.figure(figsize=(10, 6))
    val_en_transformed.plot(label="actual")
    backtest_en.plot(label="backtest (H=7)")
    plt.legend()
    plt.savefig(figure_path)
    logging.info(f"Figure saved to {figure_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TCN model on energy dataset.")
    parser.add_argument("--pretrained_path", type=str, required=True, help="Path to the pretrained TCN model.")
    parser.add_argument("--save_model_path", type=str, required=True, help="Path to save the trained TCN model.")
    parser.add_argument("--figure_path", type=str, required=True, help="Path to save the training figure.")
    parser.add_argument("--train_pkl_path", type=str, required=True, help="Path to train_en_transformed.pkl.")
    parser.add_argument("--val_pkl_path", type=str, required=True, help="Path to val_en_transformed.pkl.")
    parser.add_argument("--day_series_pkl_path", type=str, required=True, help="Path to day_series.pkl.")
    parser.add_argument("--series_en_pkl_path", type=str, required=True, help="Path to series_en.pkl.")
    
    args = parser.parse_args()
    
    train_model(args.pretrained_path, args.save_model_path, args.figure_path,
                args.train_pkl_path, args.val_pkl_path, args.day_series_pkl_path,args.series_en_pkl_path)