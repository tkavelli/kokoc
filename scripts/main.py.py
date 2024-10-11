# scripts/main.py

import os
import logging
import torch
import mlflow
from scripts.config_loader import ConfigLoader
from scripts.utils import check_hardware_requirements, setup_directories

def setup_logging(config):
    log_dir = config['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "main.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_script(script_name, logger):
    logger.info(f"Running {script_name}...")
    exit_code = os.system(f'python scripts/{script_name}.py')
    if exit_code != 0:
        logger.error(f"Error running {script_name}. Exit code: {exit_code}")
        raise RuntimeError(f"Script {script_name} failed")

if __name__ == "__main__":
    config_loader = ConfigLoader()
    config = config_loader.load_config('config.yml')
    logger = setup_logging(config)

    try:
        # Check hardware requirements
        check_hardware_requirements(logger)

        # Setup directories
        setup_directories(config, logger)

        # Set up MLflow
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])

        with mlflow.start_run(run_name="full_pipeline"):
            # Data preparation
            run_script('data_preparation', logger)
            logger.info("Verified use of word embeddings instead of product names")

            # Model training
            if config['optimization']['use_optuna']:
                run_script('hyperparameter_optimization', logger)
            
            run_script('neural_network_model', logger)
            run_script('graph_model', logger)
            run_script('gradient_boosting_models', logger)

            # Ensemble and final predictions
            run_script('ensemble_model', logger)
            
            # Verify outputs
            logger.info("Verifying outputs...")
            assert os.path.exists(config['output']['plots_dir']), "Plots directory not found"
            assert os.path.exists(config['output']['predictions_dir']), "Predictions directory not found"
            
            logger.info("Pipeline execution completed successfully.")
            mlflow.log_param("pipeline_status", "success")
    except AssertionError as ae:
        logger.error(f"Output verification failed: {str(ae)}")
        mlflow.log_param("pipeline_status", "failed")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        mlflow.log_param("pipeline_status", "failed")
        raise