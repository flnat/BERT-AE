import gc
import json
import os
import tempfile
from types import SimpleNamespace

import joblib
import lightning.pytorch as pl
import mlflow
import torch
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from src.models.autoencoder import Autoencoder, AutoEncoderConfig
from src.models.bert import BERTTrainer, BERTConfig
from src.utils.data import BERTDataModule, UnmaskedInput


def retrieve_bert_configs():
    model_config = BERTConfig.from_json_file(os.path.join(os.path.dirname(__file__), "bert_model.json"))

    with open(os.path.join(os.path.dirname(__file__), "data_config.json"), "rt") as f:
        data_config = json.load(f)

    with open(os.path.join(os.path.dirname(__file__), "bert_training.json"), "rt") as f:
        trainer_config = json.load(f, object_hook=lambda x: SimpleNamespace(**x))

    return model_config, data_config, trainer_config


def retrieve_ae_configs():
    model_config = AutoEncoderConfig.from_json_file(os.path.join(os.path.dirname(__file__), "ae_model.json"))

    with open(os.path.join(os.path.dirname(__file__), "ae_training.json"), "rt") as f:
        trainer_config = json.load(f, object_hook=lambda x: SimpleNamespace(**x))

    return model_config, trainer_config


def train_loop(run_name: str, iteration: int, iter_max: int,
               bert_model_config, data_config, bert_trainer_config,
               ae_model_config, ae_trainer_config):
    ##############################################################################################
    """
    Pretraining on MLM Task
    """
    ##############################################################################################
    pl.seed_everything(data_config["rng_seed"])

    # Instantiating the DataModule and calling the initial data preparation
    data_module = BERTDataModule(**data_config)
    data_module.prepare_data()
    # Extracting the vocab_size for building the model
    bert_model_config.vocab_size = data_module.vocab_size

    # Instantiating the model
    bert_model = BERTTrainer(bert_model_config)

    # Setting up the PL Trainer and Callbacks
    logger = MLFlowLogger(experiment_name=bert_trainer_config.experiment_name,
                          run_name=f"{run_name}-pretraining-{iteration + 1}/{iter_max}",
                          log_model=True)

    early_stopping = pl.callbacks.EarlyStopping(monitor=bert_trainer_config.early_stopping_metric,
                                                mode=bert_trainer_config.early_stopping_objective,
                                                patience=bert_trainer_config.early_stopping_patience,
                                                min_delta=bert_trainer_config.early_stopping_min_delta)
    lr_monitoring = pl.callbacks.LearningRateMonitor(log_momentum=True)

    bert_trainer = pl.Trainer(accelerator=bert_trainer_config.accelerator, max_epochs=bert_trainer_config.max_epochs,
                              callbacks=[early_stopping, lr_monitoring], logger=logger, num_sanity_val_steps=0,
                              enable_checkpointing=True, default_root_dir=bert_trainer_config.checkpoint_dir)

    # Training/Validating/Testing the model
    bert_trainer.fit(bert_model, datamodule=data_module)
    # Log additional hyperparams of the BERT Model to the mlflow run
    with mlflow.start_run(run_id=logger.run_id):
        mlflow.log_params(data_config)
        mlflow.log_params(vars(bert_trainer_config))

    # Save weights of the trained BERT Model (not the additional projection into vocab space)
    cache_dir = tempfile.mkdtemp()
    bert_ckpt = os.path.join(cache_dir, "bert.pt")
    torch.save(bert_model.bert.state_dict(), bert_ckpt)

    # GC of no longer needed objects
    del bert_trainer, bert_model, lr_monitoring, logger, early_stopping
    gc.collect()
    torch.cuda.empty_cache()
    ##############################################################################################
    """
    Finetuning on AE Architecture
    """
    ##############################################################################################
    # Retrieving datasets from old data module
    data_module.setup(None)
    vocab = data_module.vocab

    train_ds = UnmaskedInput(logs=data_module.train_ds.logs,
                             vocab=vocab)
    val_ds = UnmaskedInput(logs=data_module.val_ds.logs,
                           labels=data_module.val_ds.labels,
                           vocab=vocab)
    test_ds = UnmaskedInput(logs=data_module.test_ds.logs,
                            labels=data_module.test_ds.labels,
                            vocab=vocab)

    train_dl = DataLoader(dataset=train_ds, batch_size=data_config.batch_size, shuffle=True,
                          collate_fn=train_ds.collate, num_workers=4)
    val_dl = DataLoader(dataset=val_ds, batch_size=data_config.batch_size, shuffle=False,
                        collate_fn=val_ds.collate, num_workers=4)
    test_dl = DataLoader(dataset=test_ds, batch_size=data_config.batch_size, shuffle=False,
                         collate_fn=test_ds.collate, num_workers=4)

    del data_module
    gc.collect()
    # Setting up the PL Trainer and Callbacks
    logger = MLFlowLogger(experiment_name=ae_trainer_config.experiment_name,
                          run_name=f"{run_name}-finetuning-{iteration + 1}/{iter_max}",
                          log_model=True)

    early_stopping = pl.callbacks.EarlyStopping(monitor=ae_trainer_config.early_stopping_metric,
                                                mode=ae_trainer_config.early_stopping_objective,
                                                patience=ae_trainer_config.early_stopping_patience,
                                                min_delta=ae_trainer_config.early_stopping_min_delta
                                                )

    autoencoder = Autoencoder(config=ae_model_config, bert_config=bert_model_config.to_dict(),
                              bert_checkpoint_path=bert_ckpt)
    ae_trainer = pl.Trainer(accelerator=ae_trainer_config.accelerator, max_epochs=ae_trainer_config.max_epochs,
                            callbacks=[early_stopping], logger=logger, num_sanity_val_steps=0,
                            enable_checkpointing=True,
                            default_root_dir=ae_trainer_config.checkpoint_dir)

    ae_trainer.fit(autoencoder, train_dataloaders=train_dl, val_dataloaders=val_dl)
    ae_trainer.test(autoencoder, dataloaders=test_dl)

    with mlflow.start_run(run_id=logger.run_id):
        joblib.dump(value=vocab, filename=os.path.join(cache_dir, "vocab"))
        mlflow.log_artifact(os.path.join(cache_dir, "vocab"))
        mlflow.log_params(vars(ae_trainer_config))
        mlflow.log_params(data_config.to_dict())


if __name__ == "__main__":

    bert_model_config, data_config, bert_trainer_config = retrieve_bert_configs()
    ae_model_config, ae_trainer_config = retrieve_ae_configs()
    train_loop(run_name=f"bert_small", iteration=1, iter_max=1, bert_model_config=bert_model_config,
               data_config=data_config, bert_trainer_config=bert_trainer_config, ae_model_config=ae_model_config,
               ae_trainer_config=ae_trainer_config)
    torch.cuda.empty_cache()



