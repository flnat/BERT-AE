from typing import Any, Dict

import lightning.pytorch as pl
import numpy as np
import torch.optim
from torch.nn import functional as f

from .logbert_lm import LogBERT
from .bert import BERT


class BERTTrainer(pl.LightningModule):
    def __init__(self, config, samples_per_class: list[int] | None = None, n_specials: int | None = None):
        super().__init__()

        self.hparams.update(config.to_dict())

        self.bert = BERT(vocab_size=self.hparams.vocab_size, max_length=self.hparams.max_length,
                         n_layers=self.hparams.n_layers, hidden_size=self.hparams.hidden_size,
                         embedding_size=self.hparams.embedding_size,
                         dropout=self.hparams.dropout, attention_heads=self.hparams.attention_heads,
                         uses_temporal=self.hparams.uses_temporal, learned_positional=
                         self.hparams.learned_positional)
        self.log_bert = LogBERT(self.bert)

        self.samples_per_class = samples_per_class
        self.n_specials = n_specials

    @staticmethod
    def rebalanced_loss(logits, labels, samples_per_class: list[int], n_specials: int, beta: float):
        """
        Implementation of Class-Balanced Loss Based on Effective Number of Samples
        https://arxiv.org/abs/1901.05555

        Extended https://github.com/fcakyon/balanced-loss/blob/main/balanced_loss/losses.py for 3d Tensor Inputs
        :param logits:
        :param labels:
        :param samples_per_class:
        :param n_specials:
        :param beta:
        :return:
        """

        num_classes = logits.size(2)

        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes
        weights = torch.tensor(weights, device=logits.device).float()

        # Create additional weight vector for special tokens in vocab  --> not present in the corpus, therefore
        # their weight should be 0
        special_weights = torch.zeros(size=(n_specials,), dtype=torch.float, device=logits.device)
        weights = torch.cat([special_weights, weights])

        cb_loss = f.cross_entropy(input=logits.transpose(1, 2), target=labels, weight=weights,
                                  ignore_index=0)
        return cb_loss

    def training_step(self, batch, batch_idx):
        predictions = self.log_bert(batch.bert_tokens)
        if self.samples_per_class is not None:
            loss = self.rebalanced_loss(logits=predictions.key_prediction, labels=batch.bert_labels,
                                        samples_per_class=self.samples_per_class, n_specials=self.n_specials,
                                        beta=self.hparams.beta_balanced_loss)
        else:
            loss = f.cross_entropy(predictions.key_prediction.transpose(1, 2), batch.bert_labels, ignore_index=0)
        self.log(name="train_loss", value=loss, on_step=True, batch_size=batch.bert_tokens.size()[0])

        return loss

    def validation_step(self, batch, batch_idx):
        predictions = self.log_bert(batch.bert_tokens)
        if self.samples_per_class is not None:
            loss = self.rebalanced_loss(logits=predictions.key_prediction, labels=batch.bert_labels,
                                        samples_per_class=self.samples_per_class, n_specials=self.n_specials,
                                        beta=self.hparams.beta_balanced_loss)
        else:
            loss = f.cross_entropy(predictions.key_prediction.transpose(1, 2), batch.bert_labels, ignore_index=0)

        self.log(name="val_loss", value=loss, on_step=False, on_epoch=True, batch_size=batch.bert_tokens.size()[0])

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.bert(batch.bert_tokens)

    def configure_optimizers(self) -> Any:
        if not hasattr(self.hparams, "warmup_steps"):
            self.hparams.warmup_steps = 0.1 * self.trainer.estimated_stepping_batches

        optimizer = torch.optim.AdamW(self.log_bert.parameters(), lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay, betas=self.hparams.betas)

        return {"optimizer": optimizer}

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_closure=None) -> None:
        """
        Implement Learning Rate Scheduling as mentioned in to original BERT paper
        :param epoch:
        :param batch_idx:
        :param optimizer:
        :param optimizer_closure:
        :return:
        """
        # Warmup Logic as in huggingface
        # https://github.com/huggingface/transformers/blob/08f534d2da47875a4b7eb1c125cfa7f0f3b79642/src/transformers/optimization.py#L90
        if self.trainer.global_step < self.hparams.warmup_steps:
            scale = float(self.global_step) / float(max(1, self.hparams.warmup_steps))
        else:
            scale = max(0.0, float(self.trainer.estimated_stepping_batches - self.trainer.global_step) /
                        float(max(1, self.trainer.estimated_stepping_batches - self.hparams.warmup_steps)))

        for pg in optimizer.param_groups:
            pg["lr"] = scale * self.hparams.lr

        optimizer.step(closure=optimizer_closure)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["n_specials"] = self.n_specials
        checkpoint["samples_per_class"] = self.samples_per_class

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.n_specials = checkpoint["n_specials"]
        self.samples_per_class = checkpoint["samples_per_class"]
