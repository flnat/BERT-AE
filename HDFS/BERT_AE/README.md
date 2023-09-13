### Parameter Documentation for BERT-AE

Configuration & Hyperparameters of BERT-AE are set through the
`ae_model.json`, `ae_training.json`, `bert_model.json`, `bert_training.json`,
`data_config.json` config files

`data_config.json`
 - parsed_logs: Path to the parsed logs in .csv format, relative to project root
 - feature_column: Column containing the log sequences
 - semi_supervised: Flag, if labels are also supplied, if `False` model will be trained without validation/testing
 - labels: Path to label .csv
 - label_column: Column containing the actual labels in the table `labels`
 - join_key: Key with which to join `parsed_logs` and `labels`
 - batch_size: batch size of the Pre-Training
 - splits: Array, containing the ratios of the train/val/test split
 - split_mode: Set the split_mode {`uncontaminted`, `stratified`, `random`}, if `semi_supervised==False` only random is available
 - vocab_min_freq: Minimal frequency of a log event to be included in the vocabulary
 - vocab_max_size: Maximal size of the vocabulary
 - max_seq_length: Maximal length of an log sequence, if a given sequence is longer than `max_seq_length` it will be truncated.
 - mask_ratio: Applied masking ratio of the MLM Pre-Training
 - rng_seed: Random Seed

`bert_model.json`
 - `max_length`: Maximal supported length of a log sequence, should be >= to `max_seq_lenth`in `data_config.json`
 - `embedding_size`: Size of the input embeddings of the transformer
 - `attention_heads`: n_heads of the transformer
 - `n_layers`: Size of the transformer encoder stack
 - `hidden_size`: Size of the hidden state of the transformer
 - `learned_positional`: Flag, if `True` use a learned positional encoding, else use a sinusoidal encoding
 - `dropout`: magnitude of dropout 
 - `lr`: learning rate
 - `betas`: Array, containing the betas parameter of the `Adam` and `AdamW` Optimizers
 - `weight_decay`: Applied weight_decay/L2-Regularization

`bert_training.json`
 - max_epochs: Maximal training epochs during Pre-Training
 - enable_early_stopping: Flag to enable early_stopping, if `False` `trained_epochs == max_epochs`
 - early_stopping_metric: Which metric to use for early stopping, for the Pre-Training only `val_loss` is available
 - early_stopping_patience: Patience for early_stopping
 - early_stopping_min_delta: Minimal difference for early stopping
 - accelerator: Device on which to train, `{cpu, gpu}`
 - enable_checkpointing: Flage to enable checkpointing
 - experiment_name: Name of the associated MLFlow Experiment
 - checkpoint_dir: Directory in which the checkpoints may be saved, if not set checkpoints will be saved in the cwd

`ae_model.json`
 - input_size: Size of the Input Embeddings, must be equal to`hidden_size` from `bert_model.json`
 - hidden_size: Size of the hidden state of the first recurrent block of the AE-Encoder and AE-Decoder. Should be `> input_size`and `< code_size`
 - code_size: Size of the latent dimension.  
 - dropout: dropout of the autoencoder
 - contamination: Estimated degree of contamination, should be set to `Count of Anomalies/Count of Anomalies + Count of Normal data` 
 - cell: Type of recurrent block in the autoencoder `{lstm, gru, rnn}`
 - recon_norm: Norm of the reconstruction error to be used `{l2, l1}`
 - lr: Learning Rate
 - weight_decay: Weight Decay
 - freeze_bert: Flag, if `True` freeze the Pre-Trained Transformer Weights during Fine-Tuning
 - semi_supervised: Same as `semi_supervised` in `data_config.json`
 - plot_histogramm: Flag, if `True` enable logging of histograms of the reconstruction error

`ae_training.json`
 - max_epochs: Maximal training epochs during Fine-Tuning
 - enable_early_stopping: Flag to enable early_stopping, if `False` `trained_epochs == max_epochs`
 - early_stopping_metric: Which metric to use for early stopping, for Fine-Tuning `{val_loss, val_recall, val_precision, val_f1, val_auroc, val_average_precision}` are available
 - early_stopping_patience: Patience for early_stopping
 - early_stopping_min_delta: Minimal difference for early stopping
 - accelerator: Device on which to train, `{cpu, gpu}`
 - enable_checkpointing: Flage to enable checkpointing
 - experiment_name: Name of the associated MLFlow Experiment
 - checkpoint_dir: Directory in which the checkpoints may be saved, if not set checkpoints will be saved in the cwd
