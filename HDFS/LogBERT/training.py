import json
import os
from types import SimpleNamespace

import lightning.pytorch as pl
import mlflow
from lightning.pytorch.loggers import MLFlowLogger

from src.models.bert import LogBERTTrainer, LogBERTConfig
from src.utils.data import BERTDataModule


def retrieve_configs():
    model_config = LogBERTConfig.from_json_file(os.path.join(os.path.dirname(__file__), "model_config.json"))

    with open(os.path.join(os.path.dirname(__file__), "data_config.json"), "rt") as f:
        data_config = json.load(f)

    with open(os.path.join(os.path.dirname(__file__), "training_config.json"), "rt") as f:
        trainer_config = json.load(f, object_hook=lambda x: SimpleNamespace(**x))

    return model_config, data_config, trainer_config


if __name__ == "__main__":
    model_config, data_config, trainer_config = retrieve_configs()

    pl.seed_everything(data_config["rng_seed"])

    # Instantiating the DataModule and calling the initial data preperation
    data_module = BERTDataModule(**data_config)
    data_module.prepare_data()
    # Extracting the vocab_size for building the model
    model_config.vocab_size = data_module.vocab_size

    # Instantiating the model
    model = LogBERTTrainer(model_config)

    # Setting up the PL Trainer and Callbacks
    logger = MLFlowLogger(experiment_name=trainer_config.experiment_name,
                          log_model=True)

    early_stopping = pl.callbacks.EarlyStopping(monitor=trainer_config.early_stopping_metric,
                                                mode=trainer_config.early_stopping_objective,
                                                patience=trainer_config.early_stopping_patience,
                                                min_delta=trainer_config.early_stopping_min_delta)
    lr_monitoring = pl.callbacks.LearningRateMonitor(log_momentum=True)

    trainer = pl.Trainer(accelerator=trainer_config.accelerator, max_epochs=trainer_config.max_epochs,
                         callbacks=[early_stopping, lr_monitoring], logger=logger, num_sanity_val_steps=0,
                         enable_checkpointing=True, default_root_dir=trainer_config.checkpoint_dir)

    # Training/Validating/Testing the model
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    # Log additional hyperparams to mlflow run
    with mlflow.start_run(run_id=logger.run_id):
        mlflow.log_params(data_config)
        mlflow.log_params(vars(trainer_config))
