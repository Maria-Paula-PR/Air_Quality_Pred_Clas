import mlflow
import mlflow.keras
import tensorflow as tf
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

class MLflowCallback(tf.keras.callbacks.Callback):
    """
    Custom Keras callback for logging metrics to MLflow during training.
    This can be used with any type of model (MLP, CNN, LSTM, Conv-LSTM).
    """
    def __init__(self, log_every_n_epochs=1):
        """
        Initialize MLflow callback.
        
        Args:
            log_every_n_epochs (int): Log metrics every n epochs (default: 1)
        """
        super(MLflowCallback, self).__init__()
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch."""
        if not logs:
            return
            
        # Only log every n epochs to avoid cluttering the MLflow UI
        if (epoch + 1) % self.log_every_n_epochs == 0:
            for metric_name, metric_value in logs.items():
                mlflow.log_metric(metric_name, metric_value, step=epoch)


def setup_mlflow(experiment_name="air_quality_prediction", tracking_uri=None):
    """
    Setup MLflow tracking.
    
    Args:
        experiment_name (str): Name of the MLflow experiment
        tracking_uri (str): URI for MLflow tracking server (default: local file)
    
    Returns:
        str: Active experiment ID
    """
    # Set tracking URI if provided, otherwise use local directory
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri("file:./mlruns")
    
    # Set or create experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    return experiment_id


def start_run(model_type, run_name=None):
    """
    Start an MLflow run with appropriate tags.
    
    Args:
        model_type (str): Type of model (MLP, CNN, LSTM, Conv-LSTM)
        run_name (str, optional): Custom name for the run
    
    Returns:
        mlflow.ActiveRun: The active MLflow run
    """
    if run_name is None:
        run_name = f"{model_type}_run"
    
    return mlflow.start_run(run_name=run_name, tags={"model_type": model_type})


def log_model_params(model):
    """
    Log model architecture parameters.
    
    Args:
        model (tf.keras.Model): The Keras model to log parameters for
    """
    # Log model summary as text artifact
    summary_file = "model_summary.txt"
    with open(summary_file, 'w') as f:
        # Redirect model.summary() output to file
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    mlflow.log_artifact(summary_file)
    os.remove(summary_file)
    
    # Log number of layers and parameters
    mlflow.log_param("num_layers", len(model.layers))
    mlflow.log_param("num_parameters", model.count_params())


def log_training_params(epochs, batch_size, optimizer, learning_rate=None, **kwargs):
    """
    Log training parameters.
    
    Args:
        epochs (int): Number of epochs
        batch_size (int): Batch size
        optimizer (str or object): Optimizer used
        learning_rate (float, optional): Learning rate
        **kwargs: Additional parameters to log
    """
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    
    if isinstance(optimizer, str):
        mlflow.log_param("optimizer", optimizer)
    else:
        mlflow.log_param("optimizer", optimizer.__class__.__name__)
    
    if learning_rate is not None:
        mlflow.log_param("learning_rate", learning_rate)
    
    # Log additional parameters
    for key, value in kwargs.items():
        mlflow.log_param(key, value)


def log_performance_metrics(y_true, y_pred, prefix=""):
    """
    Log performance metrics for regression tasks.
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        prefix (str, optional): Prefix for metric names (e.g., "test_" or "val_")
    """
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred)
    except:
        mape = np.nan  # In case of division by zero
    
    # Log metrics
    mlflow.log_metric(f"{prefix}mse", mse)
    mlflow.log_metric(f"{prefix}rmse", rmse)
    mlflow.log_metric(f"{prefix}r2", r2)
    
    if not np.isnan(mape):
        mlflow.log_metric(f"{prefix}mape", mape)


def log_prediction_plot(y_true, y_pred, title="Prediction vs Actual", max_samples=200):
    """
    Create and log a prediction vs actual plot.
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        title (str): Plot title
        max_samples (int): Maximum number of samples to plot
    """
    # Limit the number of samples to plot
    if len(y_true) > max_samples:
        indices = np.arange(min(len(y_true), max_samples))
        y_true_plot = y_true[indices]
        y_pred_plot = y_pred[indices]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_true_plot, label='Actual', color='blue')
    plt.plot(y_pred_plot, label='Predicted', color='red')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    
    # Save and log the plot
    plot_path = "prediction_plot.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    os.remove(plot_path)
    plt.close()


def log_history_plot(history):
    """
    Create and log training history plots.
    
    Args:
        history (tf.keras.callbacks.History): Training history
    """
    history_dict = history.history
    
    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.plot(history_dict['loss'], label='Training Loss', color='blue')
    if 'val_loss' in history_dict:
        plt.plot(history_dict['val_loss'], label='Validation Loss', color='red')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    
    # Save and log the plot
    loss_plot_path = "loss_history.png"
    plt.savefig(loss_plot_path)
    mlflow.log_artifact(loss_plot_path)
    os.remove(loss_plot_path)
    plt.close()
    
    # Plot metrics if available (accuracy, mae, etc.)
    metrics = [m for m in history_dict.keys() if not m.startswith('val_') and m != 'loss']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        plt.plot(history_dict[metric], label=f'Training {metric}', color='blue')
        if f'val_{metric}' in history_dict:
            plt.plot(history_dict[f'val_{metric}'], label=f'Validation {metric}', color='red')
        plt.title(f'Model {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        
        # Save and log the plot
        metric_plot_path = f"{metric}_history.png"
        plt.savefig(metric_plot_path)
        mlflow.log_artifact(metric_plot_path)
        os.remove(metric_plot_path)
        plt.close()


def log_model(model, X_sample, y_sample=None):
    """
    Log the model to MLflow.
    
    Args:
        model (tf.keras.Model): The model to log
        X_sample (array): Sample input for signature inference
        y_sample (array, optional): Sample output for signature inference
    """
    if y_sample is None:
        y_sample = model.predict(X_sample[:1])
    
    signature = infer_signature(X_sample, y_sample)
    mlflow.keras.log_model(model, "model", signature=signature)


def end_run():
    """End the current MLflow run."""
    mlflow.end_run()


# Example usage in a notebook:
"""
from mlflow_utils import MLflowCallback, setup_mlflow, start_run, log_model_params, log_training_params, log_performance_metrics, log_prediction_plot, log_history_plot, log_model, end_run

# Setup MLflow
setup_mlflow(experiment_name="air_quality_prediction")

# Define model (MLP, CNN, LSTM, or Conv-LSTM)
model = tf.keras.Sequential([...])

# Start MLflow run
with start_run(model_type="MLP"):
    # Log model architecture
    log_model_params(model)
    
    # Log training parameters
    log_training_params(
        epochs=50,
        batch_size=32,
        optimizer="adam",
        learning_rate=0.001,
        validation_split=0.2
    )
    
    # Create MLflow callback
    mlflow_callback = MLflowCallback()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        ,
        verbose=1
    )
    
    # Log training history plots
    log_history_plot(history)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Log performance metrics
    log_performance_metrics(y_test, y_pred, prefix="test_")
    
    # Log prediction plot
    log_prediction_plot(y_test, y_pred, title="MLP Prediction vs Actual")
    
    # Log the model
    log_model(model, X_train, y_train)

# MLflow run ends automatically when exiting the context
"""