import logging
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import torch
import torchmetrics
import tqdm
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.collections import MetricCollection

from .config import LogBERTConfig
from .logbert_lm import LogBERT
from .bert import BERT


@dataclass
class LogBERTPrediction:
    """
    Helper class for inference output
    """
    vhm_prediction: Optional[torch.Tensor]
    mlm_prediction: Optional[torch.Tensor]
    loss_prediction: Optional[torch.Tensor]


class LogBERTTrainer(pl.LightningModule):
    def __init__(self, config: LogBERTConfig):
        super().__init__()

        # Update internal hparams dict with arguments from LogBERTConfig
        self.hparams.update(config.to_dict())

        self.bert = BERT(vocab_size=self.hparams.vocab_size, max_length=self.hparams.max_length,
                         n_layers=self.hparams.n_layers, hidden_size=self.hparams.hidden_size,
                         embedding_size=self.hparams.embedding_size,
                         dropout=self.hparams.dropout, attention_heads=self.hparams.attention_heads,
                         uses_temporal=self.hparams.uses_temporal, learned_positional=
                         self.hparams.learned_positional)
        self.log_bert = LogBERT(self.bert)

        # Set Deep-SVDD Hyperparams
        if self.hparams.use_hypersphere_loss:
            self.radius = 0
            self.center = torch.zeros((self.hparams.embedding_size,), device=self.device, dtype=torch.float)
            self.distances = []
        # Define Loss functions
        self.mask_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.hyper_criterion = nn.MSELoss()
        # Define Loss functions used line loss predictions --> Apply no batch-wise reduction
        self.mask_criterion_infer = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        self.hyper_criterion_infer = nn.MSELoss(reduction="none")

        # Define metrics for logging & validation/testing
        self.val_metrics = MetricCollection({
            "f1_score": torchmetrics.F1Score(task="binary"),
            "precision": torchmetrics.Precision(task="binary"),
            "recall": torchmetrics.Recall(task="binary"),
            "average_precision": torchmetrics.AveragePrecision(task="binary"),
            "auroc": torchmetrics.AUROC(task="binary")
        }, prefix="val_")

        self.test_metrics = MetricCollection({
            "f1_score": torchmetrics.F1Score(task="binary"),
            "precision": torchmetrics.Precision(task="binary"),
            "recall": torchmetrics.Recall(task="binary"),
            "average_precision": torchmetrics.AveragePrecision(task="binary"),
            "auroc": torchmetrics.AUROC(task="binary")
        }, prefix="test_")

        self.roc_display = torchmetrics.ROC(task="binary")
        self.pr_rc_display = torchmetrics.PrecisionRecallCurve(task="binary")

        self.save_hyperparameters(ignore=["bert_model", "bert", "config"])

        self.test_label_cache = []
        # Manual Optimization due to LR Scheduling
        # self.automatic_optimization = False

    def forward(self, x):
        return self.log_bert.forward(x, time_info=None)

    def __call__(self, x):
        return self.forward(x)

    def training_step(self, batch, batch_idx):

        if self.hparams.use_hypersphere_loss:
            loss, masked_loss, hyper_loss, distances = self.loss_iteration(batch)
            self.log("combined_loss_train", loss)
            self.log("masked_loss_train", masked_loss)
            self.log("hyper_loss_train", hyper_loss)

            # Write observed distances into cache for later radius calculation
            self.distances.append(distances)
        else:
            loss = self.loss_iteration(batch)
            self.log("masked_loss_train", loss)

        # # Optimization setup
        # optimizers = self.optimizers()
        # # lr_scheduler = self.lr_schedulers()
        #
        # # Backprop with lr-scheduling
        # optimizers.optimizer.zero_grad()
        # self.manual_backward(loss)
        # optimizers.optimizer.step()
        # lr_scheduler.step()

        return loss

    def on_train_epoch_start(self) -> None:
        if self.hparams.use_hypersphere_loss:
            # Before training, we will initialize the center of the <DIST> representations
            console_logger = logging.getLogger("lightning")
            # Reinitialize train_loader because lightning is wierd
            console_logger.info("Calculating Sphere Center of CLS-Representation of Training Data")
            # self.trainer.fit_loop.setup_data()
            dl = self.trainer.train_dataloader

            self.calculate_center(dl)
            console_logger.info("Finished calculating sphere center of CLS-Representation")

            center_norm = torch.linalg.vector_norm(self.center)

            console_logger.info(f"Norm of center representation: {center_norm.item():.4f}")

    def on_train_epoch_end(self) -> None:
        if self.hparams.use_hypersphere_loss:
            self.radius = self.get_radius(torch.concatenate(self.distances, dim=0))
            self.radius = self.radius.to(self.device)
            # Clear distance cache
            self.distances.clear()

    def validation_step(self, batch, batch_idx):
        if self.hparams.use_hypersphere_loss:
            loss, masked_loss, hyper_loss, _ = self.loss_iteration(batch)
            self.log("combined_loss_val", loss, on_step=False, on_epoch=True)
            self.log("masked_loss_val", masked_loss, on_step=False, on_epoch=True)
            self.log("hyper_loss_val", hyper_loss, on_step=False, on_epoch=True)

        else:
            loss = self.loss_iteration(batch)
            self.log("masked_loss_val", loss, on_step=False, on_epoch=True)

        predictions = self.predict_iteration(batch)
        predictions = self.choose_prediction(predictions)
        self.val_metrics.update(predictions, batch.sequence_labels)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        predictions = self.predict_iteration(batch)
        predictions = self.choose_prediction(predictions)

        self.test_metrics.update(predictions, batch.sequence_labels)
        self.pr_rc_display.update(predictions, batch.sequence_labels)
        self.roc_display.update(predictions, batch.sequence_labels)

        self.test_label_cache.append(batch.sequence_labels.cpu().detach().numpy())
    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)
        self.test_metrics.reset()

        # Calculated Prevalence of pos class
        class_count = Counter(np.concatenate(self.test_label_cache))
        prevalence_pos_label = class_count[1] / sum(class_count.values())
        # Introspect experiment and run_id to enable manual logging of figures
        logger = self.logger.experiment
        run_id = self.logger.run_id

        # Compute TPR & FPR for ROC Display
        tpr, fpr, _ = self.roc_display.compute()
        self.roc_display.reset()
        auc = self.trainer.logged_metrics["test_auroc"]

        # Compute Precision-Recall Display
        precision, recall, _ = self.pr_rc_display.compute()
        self.pr_rc_display.reset()
        ap = self.trainer.logged_metrics["test_average_precision"]

        roc_plot = self.plot_roc(tpr.cpu(), fpr.cpu(), auc, prevalence_pos_label)
        logger.log_figure(run_id, figure=roc_plot, artifact_file="roc.pdf")

        pr_rc_plot = self.plot_precision_recall(precision.cpu(), recall.cpu(), ap, prevalence_pos_label)
        logger.log_figure(run_id, figure=pr_rc_plot, artifact_file="pr_rc.pdf")

        self.test_label_cache.clear()
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        return self.predict_iteration(batch)

    def configure_optimizers(self):

        # Calculate size of warmup scheduling of warmup_step hyperparameter is not specified
        # Use 10 % of training as warmup as a default

        if not hasattr(self.hparams, "warmup_steps"):
            self.hparams.warmup_steps = 0.1 * self.trainer.estimated_stepping_batches

        optimizer = optim.AdamW(self.log_bert.parameters(),
                                lr=self.hparams.lr, betas=self.hparams.betas,
                                weight_decay=self.hparams.weight_decay)

        return {"optimizer": optimizer}

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer,
            optimizer_closure=None,
    ) -> None:
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

    def loss_iteration(self, batch, predict_mode: bool = False):
        """Basic logic for loss calculation on a single batch, wrapped by train_step and val_step"""
        # predict_mode disable dropout if used for inference in line loss approach

        result = self.forward(batch.bert_tokens)
        masked_loss = self.mask_criterion(result.key_prediction.transpose(1, 2), batch.bert_labels)
        if self.hparams.use_hypersphere_loss:
            if self.center.get_device() == -1:
                self.center = self.center.to(self.device)
            hyper_loss = self.hyper_criterion(result.cls_representation.squeeze(),
                                              self.center.expand(batch.bert_tokens.shape[0], -1))

            combined_loss = masked_loss + self.hparams.alpha * hyper_loss
            distances = torch.sum((result.cls_representation - self.center) ** 2, dim=1).cpu()
            return combined_loss, masked_loss, hyper_loss, distances

        return masked_loss

    def predict_iteration(self, batch):
        # Basic logic for running inference on sing batch, wrapped by predict_step and test_step
        result = self(batch.bert_tokens)

        # masked_lm_output: (batch_size, sequence_length, vocab_size)
        # cls_output: (batch_size, hidden_size) --> 2D because we are only accessing the <CLS> Representation
        # bert_label: (batch_size, sequence_length)

        masked_lm_output, cls_output = result.key_prediction, result.cls_representation

        mlm_prediction = self.mlm_prediction(key_predictions=masked_lm_output, masked_labels=batch.bert_labels)
        loss_prediction = self.loss_prediction(batch=batch)
        if self.hparams.use_hypersphere_loss:
            vhm_prediction = self.vhm_prediction(cls_representation=cls_output)

            return LogBERTPrediction(vhm_prediction=vhm_prediction, mlm_prediction=mlm_prediction,
                                     loss_prediction=loss_prediction)
        return LogBERTPrediction(mlm_prediction=mlm_prediction, vhm_prediction=None, loss_prediction=loss_prediction)

    def calculate_center(self, data_loader: DataLoader) -> None:
        """
        Calculate the Center of the Hypersphere used for LogBERTs VHM Loss
        Called after each training epoch (after the tenth epoch) to update the center representation
        :param data_loader: DataLoader of the TrainingSet
        :return: Center
        """
        center = torch.zeros((self.hparams.embedding_size,), dtype=torch.float, device=self.device)
        # Disable gradient computation
        batch_count = 0
        with torch.no_grad():
            # Disable dropout
            self.log_bert.train(mode=False)

            loader = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
            for idx, batch in loader:
                result = self(batch.bert_tokens.to(self.device))
                cls_output = result.cls_representation

                center += torch.sum(cls_output.detach().to(self.device), dim=0)
                batch_count += batch.bert_tokens.shape[0]
        # Get the mean representation of the CLS Tokens
        self.center = center / batch_count

        # re-enable dropout
        self.log_bert.train(mode=True)

    def get_radius(self, dist: torch.Tensor) -> torch.Tensor:
        return torch.quantile(torch.sqrt(dist), 1 - self.hparams.nu)

    def mlm_prediction(self, key_predictions: torch.Tensor, masked_labels: torch.Tensor) -> torch.Tensor:
        """
        Implement the Candidate-Set approach to detect Anomalies using MLM-Task Output e.g. if the masked token is
        not in the top [self.num_candidates] activations of the MLM-Task, then the token is to be considered a
        miss-prediction. If the number of miss-predictions in given sequence is above an a-priori specified threshold
        (typically 50 % of masked tokens), then the sequence is to be considered anomalous.
        :param key_predictions: MLM-Output in shape (batch, seq_len, vocab_len)
        :param masked_labels: Masked Labels in shape (batch, seq_len)
        :return: percentage of misspredictions per sequence in shape (batch,)
        """

        # Original implementation didn't use vectorized logic to find miss-predictions, and relied instead on nested
        # loops. As there are reports on performance issues during inference
        # (https://github.com/HelenGuohx/logbert/issues/40), we will implement vectorized logic from the get-go.

        # We are only interested in the models ability to successfully predict masked tokens, therefore we will
        # filter out unmasked tokens (unmasked tokens were labelled with 0)
        masked_idx = (masked_labels > 0).int()
        # Calculate how many tokens have been masked per sequence
        masked_count = torch.sum(masked_idx, dim=-1)

        # Find the set of the n most likely tokens
        candidate_set = torch.argsort(key_predictions, dim=-1, descending=True)[:, :, :self.hparams.num_candidates]

        miss_predictions = ~((candidate_set == masked_labels.unsqueeze(-1)).any(-1))

        miss_predictions = torch.sum(miss_predictions * masked_idx, dim=-1)

        # Return miss-classified masked tokens relative to total masked tokens
        return torch.nan_to_num(miss_predictions / masked_count)

    def vhm_prediction(self, cls_representation: torch.FloatTensor) -> torch.IntTensor:
        """
        Performs Inference using the VHM Approach
        :param cls_representation:
        :return:
        """
        distances = torch.sqrt(torch.sum((cls_representation - self.center) ** 2, dim=-1))

        return (distances > self.radius).int()

    def loss_prediction(self, batch) -> torch.FloatTensor:
        """
        Performs inference using the Line Loss Approach --> if the loss of a given sample is above a threshold it
        is anomalous.
        :param batch: LogBERTInput Wrapper Class
        :return: Unreduced Loss
        """
        self.log_bert.train(mode=False)

        result = self.forward(batch.bert_tokens)
        loss = self.mask_criterion_infer(result.key_prediction.transpose(1, 2), batch.bert_labels)
        # Because we get loss in shape (batch, length_seq) we need to reduce along the second axis
        loss = torch.mean(loss, dim=-1)
        if self.hparams.use_hypersphere_loss:
            if self.center.get_device() == -1:
                # if center tensor is on cpu, transfer it to the current device --> probably gpu
                self.center = self.center.to(self.device)
            hyper_loss = self.hyper_criterion_infer(result.cls_representation.squeeze(),
                                                    self.center.expand(batch.bert_tokens.shape[0], -1))
            hyper_loss = torch.mean(hyper_loss, dim=-1)

            loss = loss + self.hparams.alpha * hyper_loss
        self.log_bert.train(mode=True)
        return loss

    def choose_prediction(self, predictions: LogBERTPrediction):
        """
        Utility Method to enable dynamic switching of logged prediction method
        :param predictions:
        :return:
        """
        match self.hparams.prediction_method:
            case "loss":
                return predictions.loss_prediction
            case "vhm":
                return predictions.vhm_prediction
            case "mlm":
                return predictions.mlm_prediction
            case _:
                raise ValueError("Predictions must be either of {loss, vhm, mlm}!")

    @staticmethod
    def plot_roc(tpr, fpr, auc, prevalence_pos_label):
        plt.clf()
        ax = plt.subplot()
        ax.plot(tpr, fpr, label=f"ROC Curve (area={auc.item():.2f})")
        ax.plot((0, 1), (0, 1), label="Chance level (AUC = 0.5)", color="k", linestyle="--")
        ax.set_xlabel("False Positive Rate (Positive label: 1)")
        ax.set_ylabel("True Positive Rate (Positive label: 1)")
        # ax.set_title("Receiver operating characteristic")
        ax.legend(loc="best")

        return plt.gcf()

    @staticmethod
    def plot_precision_recall(precision, recall, average_precision, prevalence_pos_label):
        plt.clf()
        ax = plt.subplot()

        ax.plot(recall, precision, label=f"AP = {average_precision:.2f}")
        ax.plot((0, 1), (prevalence_pos_label, prevalence_pos_label),
                label=f"Chance level (AP = {prevalence_pos_label:0.2f})",
                color="k", linestyle="--")

        ax.set_xlabel("Recall (Positive label: 1)")
        ax.set_ylabel("Precision (Positive label: 1)")
        # ax.set_title("Precision-Recall Curve")
        ax.legend(loc="best")

        return plt.gcf()
