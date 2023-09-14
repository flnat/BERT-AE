### Parameter Documentation for BERT-AE

Configuration & Hyperparameters of BERT-AE are set through the `model_config.json`, `training_config.json` and
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

`model_config.json`
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
 - `use_hypersphere_loss`: Flag, if `True` model is trained with a combined MLM & Deep-SVDD Objective as specified in the original LogBERT paper
 - `alpha`: Hyperparameter, controlling the impact of the Deep-SVDD task
 - `nu`: nu parameter for the Deep-SVDD Objective, can be seen analogous to the nu param in the Nu-SVM Parametrization i.e. sets an upper bound of the ratio of support vectors relative to all data points 

`training_config.json`
 - max_epochs: Maximal training epochs during Pre-Training
 - enable_early_stopping: Flag to enable early_stopping, if `False` `trained_epochs == max_epochs`
 - early_stopping_metric: Which metric to use for early stopping
 - early_stopping_patience: Patience for early_stopping
 - early_stopping_min_delta: Minimal difference for early stopping
 - accelerator: Device on which to train, `{cpu, gpu}`
 - enable_checkpointing: Flage to enable checkpointing
 - experiment_name: Name of the associated MLFlow Experiment
 - checkpoint_dir: Directory in which the checkpoints may be saved, if not set checkpoints will be saved in the cwd
