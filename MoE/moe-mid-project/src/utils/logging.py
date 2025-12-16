import logging

def setup_logging(log_file='moe_mid_project.log', level=logging.INFO):
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=level
    )

def log_experiment_info(experiment_name, params):
    logging.info(f"Experiment: {experiment_name}")
    logging.info(f"Parameters: {params}")

def log_metrics(metrics):
    for key, value in metrics.items():
        logging.info(f"{key}: {value}")

def log_error(message):
    logging.error(message)