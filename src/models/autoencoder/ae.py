from typing import Dict, Any, Optional

import lightning.pytorch as pl
import matplotlib.pyplot
import numpy as np
import seaborn as sns
import torch
import tqdm
from matplotlib import pyplot as plt
from sklearn import metrics
from torch import nn, optim
from torch.nn import functional as f
from torch.utils.data import DataLoader

from .config import AutoEncoderConfig
from ..bert import BERT


class Encoder(nn.Module):
    """
    Encoder Component for the Autoencoder, currently hard coded to only use two stacked recurrent layers. For larger stacks
    we'd probably use stronger hardware & introduce Residual Connections.
    """
    def __init__(self, input_size: int, hidden_size: int, code_size: int,
                 dropout_rate: float, cell: str):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.code_size = code_size
        self.dropout_rate = dropout_rate
        self.cell_type = cell
        match cell:
            case "gru":
                self.rnn1 = nn.GRU(input_size, hidden_size, batch_first=True)
                self.rnn2 = nn.GRU(hidden_size, code_size, batch_first=True)
                self.dropout = nn.Dropout(dropout_rate)
            case "lstm":
                self.rnn1 = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.rnn2 = nn.LSTM(hidden_size, code_size, batch_first=True)
                self.dropout = nn.Dropout(dropout_rate)
            case "rnn":
                self.rnn1 = nn.RNN(input_size, hidden_size, batch_first=True)
                self.rnn2 = nn.RNN(hidden_size, code_size, batch_first=True)
                self.dropout = nn.Dropout(dropout_rate)
            case _:
                raise ValueError("Cell must be either of {rnn, gru, lstm}")

    def forward(self, x):
        x, _ = self.rnn1(x)
        x = self.dropout(x)
        _, state = self.rnn2(x)

        if self.cell_type == "lstm":
            # Incase of LSTM Layers return only the hidden state
            return state[0]
        else:
            return state


class Decoder(nn.Module):
    """
    Decoder Component for Autoencoder, currently hard coded to only use two stacked recurrent layers. For larger stacks
    we'd probably use stronger hardware & introduce Residual Connections.
    """
    def __init__(self, input_size: int, hidden_size: int, code_size: int,
                 dropout_rate: float, cell: str):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.code_size = code_size
        self.dropout_rate = dropout_rate
        self.cell_type = cell
        match cell:
            case "gru":
                self.rnn1 = nn.GRU(input_size, hidden_size, batch_first=True)
                self.rnn2 = nn.GRU(hidden_size, code_size, batch_first=True)
                self.dropout = nn.Dropout(dropout_rate)
            case "lstm":
                self.rnn1 = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.rnn2 = nn.LSTM(hidden_size, code_size, batch_first=True)
                self.dropout = nn.Dropout(dropout_rate)
            case "rnn":
                self.rnn1 = nn.RNN(input_size, hidden_size, batch_first=True)
                self.rnn2 = nn.RNN(hidden_size, code_size, batch_first=True)
                self.dropout = nn.Dropout(dropout_rate)
            case _:
                raise ValueError("Cell must be either of {rnn, gru, lstm}")

    def forward(self, x, sequence_length: int):
        x = torch.transpose(x, 0, 1)
        x = x.expand(-1, sequence_length, -1)

        x, _ = self.rnn1(x)
        x = self.dropout(x)
        x, _ = self.rnn2(x)

        return x


class Autoencoder(pl.LightningModule):
    """
    The Autoencoder implements a Seq2Seq Architecture, we therefore use the final hidden state of the encoders LSTM stack
    as the latent representation.
    """
    def __init__(self, bert_config: dict, config: AutoEncoderConfig,
                 bert_checkpoint_path: Optional[str] = None, from_checkpoint: bool = False):
        super().__init__()
        self.bert = BERT.from_config(bert_config)

        if bert_checkpoint_path is not None and not from_checkpoint:
            # If no path to pre-trained BERT weights, just freshly initialize the BERT Model
            # useful if loading already fine-tuned checkpoint --> no point in loading in pre-trained BERT Embeddings

            bert_state_dict = torch.load(bert_checkpoint_path)
            self.bert.load_state_dict(bert_state_dict)

        self.hparams.update(config.to_dict())
        self.config = config

        if self.hparams.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.encoder = Encoder(self.hparams.input_size, self.hparams.hidden_size, self.hparams.code_size,
                               dropout_rate=self.hparams.dropout, cell=self.hparams.cell)
        self.decoder = Decoder(self.hparams.code_size, self.hparams.hidden_size, self.hparams.input_size,
                               dropout_rate=self.hparams.dropout, cell=self.hparams.cell)
        # Instantiate loss function
        match self.hparams.recon_norm:
            case "l2":
                self.recon_loss = nn.MSELoss()
            case "l1":
                self.recon_loss = nn.L1Loss()
        # Initialization of additional hyperparameters
        self.max_score = np.NINF
        self.min_score = np.Inf
        self.thresh = 0  # Initialize to 0 due to sanity checking

        # Cache Lists for batched predictions in validation and testing
        self.cache_train_err = []

        self.cache_val_err = []
        self.cache_val_lbl = []

        self.cache_test_err = []
        self.cache_test_lbl = []
        self.save_hyperparameters(ignore=["config"])

    def forward(self, x, sequence_length):
        x = self.bert(x)
        z = self.encoder(x)
        z = self.decoder(z, sequence_length)

        return z

    def training_step(self, batch, batch_idx):
        sequence_length = batch.size(1)

        bert_representations = self.bert(batch)
        reconstructions = self.forward(batch, sequence_length)
        loss = self.recon_loss(reconstructions, bert_representations)

        self.log("train_loss", loss, prog_bar=True, on_step=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """
        Plot the Histogramm of the observed reconstruction errors of the train-set to get a rough outline
        of their distribution --> might be relevant for more advanced thresholding & pretty pictures in thesis.
        :return:
        """
        if self.hparams.plot_histogramm:
            logger = self.logger.experiment
            run_id = self.logger.run_id

            hist = self.plot_recon_error_hist(torch.cat(self.cache_train_err),
                                              threshold=self.thresh)
            logger.log_figure(run_id, figure=hist,
                              artifact_file=f"recon_errors_hist/train/epoch_{self.trainer.current_epoch}.pdf")
            plt.close(hist)
            self.cache_train_err.clear()

    def on_validation_epoch_start(self) -> None:
        """
        Lightning uses a completely unintuitive ordering of its hooks, whereby on_train_epoch_end gets called after
        the validation logic. Because this would lead to wrong thresholds being used, we will have to move the threshold
        calculation to the on_validation_epoch_start hook.
        It's important to note that we will still only use the training data for calculating the threshold.
        """
        self.calculate_threshold()

    def validation_step(self, batch, batch_idx):
        if self.hparams.semi_supervised:
            batch, labels = batch

        sequence_length = batch.size(1)

        bert_representation = self.bert(batch)
        recons = self(batch, sequence_length)
        loss = self.recon_loss(recons, bert_representation)
        self.log("val_loss", loss)
        if self.hparams.semi_supervised:
            reconstruction_errors = self.predict_iteration(batch)
            self.cache_val_err.append(reconstruction_errors.detach().cpu().numpy())
            self.cache_val_lbl.append(labels.cpu().numpy())

    def on_validation_epoch_end(self) -> None:
        if self.hparams.semi_supervised:
            recon_errors = np.concatenate(self.cache_val_err)
            labels = np.concatenate(self.cache_val_lbl)

            preds = (recon_errors > self.thresh).astype("int")
            self.log("val_recall", metrics.recall_score(labels, preds), on_epoch=True, on_step=False)
            self.log("val_precision", metrics.precision_score(labels, preds), on_epoch=True, on_step=False)
            self.log("val_f1", metrics.f1_score(labels, preds), on_epoch=True, on_step=False)
            self.log("val_auroc", metrics.roc_auc_score(labels, recon_errors), on_epoch=True, on_step=False)
            self.log("val_average_precision", metrics.average_precision_score(labels, recon_errors), on_epoch=True,
                     on_step=False)

            # Introspect experiment and run_id to enable manual logging of figures
            logger = self.logger.experiment
            run_id = self.logger.run_id

            roc = metrics.RocCurveDisplay.from_predictions(y_true=labels, y_pred=recon_errors, plot_chance_level=True)
            logger.log_figure(run_id, figure=roc.figure_,
                              artifact_file=f"roc/val/epoch_{self.trainer.current_epoch}.pdf")

            pr_rc = metrics.PrecisionRecallDisplay.from_predictions(y_true=labels, y_pred=recon_errors,
                                                                    plot_chance_level=True)
            logger.log_figure(run_id, figure=pr_rc.figure_,
                              artifact_file=f"pr_rc/val/epoch_{self.trainer.current_epoch}.pdf")

            # Recon Errors histogramm
            hist = self.plot_recon_error_hist(recon_errors,
                                              threshold=self.thresh)
            logger.log_figure(run_id, figure=hist,
                              artifact_file=f"recon_errors_hist/val/epoch_{self.trainer.current_epoch}.pdf")
            plt.close(hist)


            # Close MPL figures
            plt.close(roc.figure_)
            plt.close(pr_rc.figure_)
            # Clean up cache lists
            self.cache_val_lbl.clear()
            self.cache_val_err.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if self.hparams.semi_supervised:
            batch, labels = batch
            reconstruction_errors = self.predict_iteration(batch)
            self.cache_test_err.append(reconstruction_errors.detach().cpu().numpy())
            self.cache_test_lbl.append(labels.cpu().numpy())

    def on_test_epoch_end(self) -> None:
        if self.hparams.semi_supervised:
            recon_errors = np.concatenate(self.cache_test_err)
            labels = np.concatenate(self.cache_test_lbl)

            preds = (recon_errors > self.thresh).astype("int")

            self.log("test_recall", metrics.recall_score(labels, preds), on_epoch=True, on_step=False)
            self.log("test_precision", metrics.precision_score(labels, preds), on_epoch=True, on_step=False)
            self.log("test_f1", metrics.f1_score(labels, preds), on_epoch=True, on_step=False)
            self.log("test_auroc", metrics.roc_auc_score(labels, recon_errors), on_epoch=True, on_step=False)
            self.log("test_average_precision", metrics.average_precision_score(labels, recon_errors), on_epoch=True,
                     on_step=False)

            # Introspect experiment and run_id to enable manual logging of figures
            logger = self.logger.experiment
            run_id = self.logger.run_id

            roc = metrics.RocCurveDisplay.from_predictions(y_true=labels, y_pred=recon_errors, plot_chance_level=True)
            logger.log_figure(run_id, figure=roc.figure_, artifact_file="roc/test/roc_test.pdf")

            pr_rc = metrics.PrecisionRecallDisplay.from_predictions(y_true=labels, y_pred=recon_errors,
                                                                    plot_chance_level=True)
            logger.log_figure(run_id, figure=pr_rc.figure_, artifact_file="pr_rc/test/pr_rc_test.pdf")
            # Close mpl figures
            plt.close(roc.figure_)
            plt.close(pr_rc.figure_)
            # Recon Errors histogramm
            hist = self.plot_recon_error_hist(recon_errors,
                                              threshold=self.thresh)
            logger.log_figure(run_id, figure=hist,
                              artifact_file=f"recon_errors_hist/test/epoch_{self.trainer.current_epoch}.pdf")
            plt.close(hist)

            # Clean up cache lists
            self.cache_test_lbl.clear()
            self.cache_test_err.clear()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.predict_iteration(batch)

    def configure_optimizers(self):

        # Calculate size of warmup scheduling of warmup_step hyperparameter is not specified
        # Use 10 % of training as warmup as a default

        if not hasattr(self.hparams, "warmup_steps"):
            self.hparams.warmup_steps = 0.1 * self.trainer.estimated_stepping_batches

        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr, betas=self.hparams.betas,
                                weight_decay=self.hparams.weight_decay)

        return {"optimizer": optimizer}

    # def optimizer_step(
    #         self,
    #         epoch: int,
    #         batch_idx: int,
    #         optimizer,
    #         optimizer_closure=None,
    # ) -> None:
    #     """
    #     Implement Learning Rate Scheduling as mentioned in to original BERT paper
    #     :param epoch:
    #     :param batch_idx:
    #     :param optimizer:
    #     :param optimizer_closure:
    #     :return:
    #     """
    #     # Warmup Logic as in huggingface
    #     # https://github.com/huggingface/transformers/blob/08f534d2da47875a4b7eb1c125cfa7f0f3b79642/src/transformers/optimization.py#L90
    #     if self.trainer.global_step < self.hparams.warmup_steps:
    #         scale = float(self.global_step) / float(max(1, self.hparams.warmup_steps))
    #     else:
    #         scale = max(0.0, float(self.trainer.estimated_stepping_batches - self.trainer.global_step) /
    #                     float(max(1, self.trainer.estimated_stepping_batches - self.hparams.warmup_steps)))
    #
    #     for pg in optimizer.param_groups:
    #         pg["lr"] = scale * self.hparams.lr
    #
    #     optimizer.step(closure=optimizer_closure)

    def predict_iteration(self, batch, normalized: bool = True, reduce_batch: bool = True):
        """
        Common Logic for predictions. Used in validation, testing, prediction steps
        :param batch: batch of Log-Tokens as LongIntegers in shape (n_batches, len_seq)
        :param normalized: Whether to min max scale the recon error
        :param reduce_batch: Whether to reduce to shape (n_batches, ) with torch.mean
        :return: Reconstruction-errors in shape (n_batches,) or (n_batches, len_seq)
        """
        sequence_length = batch.size(1)

        bert_representations = self.bert(batch)
        reconstructions = self.forward(batch, sequence_length)

        match self.hparams.recon_norm:
            case "l2":
                recon_error = f.mse_loss(reconstructions, bert_representations, reduction="none")
            case "l1":
                recon_error = f.l1_loss(reconstructions, bert_representations, reduction="none")
        if reduce_batch:
            recon_error = torch.mean(recon_error, dim=(1, 2)).detach()
        else:
            recon_error = torch.mean(recon_error, dim=(2)).detach()

        if normalized:
            recon_error = (recon_error - self.min_score) / (self.max_score - self.min_score)
        return recon_error

    def predict_unreduced(self, predict_dl: DataLoader) -> list[torch.Tensor]:
        errors = []
        # Disable dropout
        self.eval()
        print("Calculating unreduced reconstruction errors for specified data:")
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(predict_dl)):
                batch = batch.to(self.device)

                errors.append(self.predict_iteration(batch, normalized=True, reduce_batch=False).detach().cpu())

        # Reenable dropo
        self.train()
        return errors

    def calculate_threshold(self):
        """
        Performs an additional forward pass on the entire training set and records both the highest and the lowest
        reconstruction error, --> later used for normalization of scores for easier/prettier thresholding of scores
        :return: None
        """
        print(f"Retrieving max and min observed reconstruction error for epoch {self.trainer.current_epoch}")

        # Reset Scores for clean normalization
        self.max_score = np.NINF
        self.min_score = np.inf

        scores = []
        self.trainer.fit_loop.setup_data()
        dl = self.trainer.train_dataloader

        for batch in dl:
            # Disable gradient computation
            with torch.no_grad():
                if isinstance(batch, tuple):
                    batch, _ = batch

                sequence_length = batch.size(1)
                # Move batch to device used by the model
                batch = batch.to(self.device)

                bert_representations = self.bert(batch)
                reconstructed_embeddings = self.forward(batch, sequence_length)

                match self.hparams.recon_norm:
                    case "l2":
                        reconstruction_error = f.mse_loss(reconstructed_embeddings, bert_representations,
                                                          reduction="none")
                    case "l1":
                        reconstruction_error = f.l1_loss(reconstructed_embeddings, bert_representations,
                                                         reduction="none")
                # Reduce reconstruction-error on batch dim
                reconstruction_error = torch.mean(reconstruction_error, dim=(1, 2)).detach()
                self.cache_train_err.append(reconstruction_error.detach().cpu())
                # scores.append(reconstruction_error)

                min_batch = torch.min(reconstruction_error)
                max_batch = torch.max(reconstruction_error)

                if max_batch > self.max_score:
                    self.max_score = max_batch
                if min_batch < self.min_score:
                    self.min_score = min_batch

        print(f"max reconstruction error: {self.max_score:.4f}")
        print(f"min reconstruction error: {self.min_score:.4f}")

        self.thresh = torch.quantile(torch.cat(self.cache_train_err), q=1 - self.hparams.contamination).item()
        # Rescale percentile into interval [0, 1] using min max scaling
        self.thresh = ((self.thresh - self.min_score) / (
                self.max_score - self.min_score)).item()
        print(
            f"Estimated {(1 - self.hparams.contamination) * 100}th percentile of normalized reconstruction errors is "
            f"at: {self.thresh:.4f}")

        self.log("max_reconstruction_loss", self.max_score, on_epoch=True, on_step=False)
        self.log("min_reconstruction_loss", self.min_score, on_epoch=True, on_step=False)
        self.log("threshold", self.thresh, on_epoch=True, on_step=False)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["max_score"] = self.max_score
        checkpoint["min_score"] = self.min_score
        checkpoint["thresh"] = self.thresh
        checkpoint["config"] = self.config
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.max_score = checkpoint["max_score"]
        self.min_score = checkpoint["min_score"]
        self.thresh = checkpoint["thresh"]
        self.config = checkpoint["config"]
    @staticmethod
    def plot_roc(tpr, fpr, auc):
        plt.clf()
        ax = plt.subplot()
        ax.plot(tpr, fpr, label=f"ROC Curve (area={auc.item():.2f})")
        ax.plot(tpr, tpr, label=f"Random Guessing", c="tab:red", ls="--")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.05, 1.05)

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver operating characteristic")
        ax.legend(loc=4)

        return plt.gcf()

    @staticmethod
    def plot_precision_recall(precision, recall, average_precision):
        plt.clf()
        ax = plt.subplot()
        ax.plot(recall, precision, label=f"AP = {average_precision:.2f}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.05, 1.05)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="best")

        return plt.gcf()

    @staticmethod
    def plot_recon_error_hist(recon_error, threshold: Optional[float] = None) -> matplotlib.figure.Figure:
        plt.clf()

        ax = plt.subplot()
        sns.histplot(recon_error, log_scale=False, ax=ax)
        if threshold is not None:
            plt.axvline(threshold, c="tab:red", linestyle="--", label=f"Schwellwert: {threshold:.4f}")
            ax.set_xlabel("Normalisierter Rekonstruktionsfehler")
            ax.set_ylabel("Frequenz")
            ax.set_yscale('log')
            ax.legend()
        # ax.set_title("Histogramm of unnormalized Reconstruction Errors")

        return plt.gcf()
