import math
from typing import Optional, Dict, Union
from functools import partial

from htr.models.fphtr.fphtr import FullPageHTREncoderDecoder
from htr.models.sar.sar import ShowAttendRead
from htr.util import LabelEncoder

import torch.optim as optim
import pytorch_lightning as pl
from torch import Tensor


class LitShowAttendRead(pl.LightningModule):
    model: ShowAttendRead

    """
    Pytorch Lightning module that acting as a wrapper around the ShowAttendRead class.

    Using a PL module allows the model to be used in conjunction with a Pytorch
    Lightning Trainer, and takes care of logging relevant metrics to Tensorboard.
    """

    def __init__(
        self,
        label_encoder: LabelEncoder,
        learning_rate: float = 0.001,
        max_seq_len: int = 50,
        d_enc: int = 512,
        # d_model: int = 512,
        d_k: int = 512,
        resnet_arch: str = "resnet31",
        dec_dropout: int = 0.0,
        enc_dropout: int = 0.1,
        pred_dropout: int = 0.1,
        use_lr_scheduler: bool = False,
        loss_reduction: str = "mean",
        vocab_len: Optional[int] = None,  # if not specified len(label_encoder) is used
        params_to_log: Optional[Dict[str, Union[str, float, int]]] = None,
    ):
        super().__init__()

        # Save hyperparameters.
        self.learning_rate = learning_rate
        self.use_lr_scheduler = use_lr_scheduler
        if params_to_log is not None:
            self.save_hyperparameters(params_to_log)
        self.save_hyperparameters(
            "max_seq_len",
            "learning_rate",
            "d_enc",
            # "d_model",
            "resnet_arch",
            "use_lr_scheduler",
            "dec_dropout",
            "enc_dropout",
            "pred_dropout",
            "d_k",
        )

        # Initialize the model.
        self.model = ShowAttendRead(
            label_encoder=label_encoder,
            max_seq_len=max_seq_len,
            d_enc=d_enc,
            # d_model=d_model,
            resnet_arch=resnet_arch,
            dec_dropout=dec_dropout,
            enc_dropout=enc_dropout,
            pred_dropout=pred_dropout,
            d_k=d_k,
            loss_reduction=loss_reduction,
            vocab_len=vocab_len,
        )

    @property
    def resnet_encoder(self):
        return self.model.resnet_encoder

    @property
    def lstm_decoder(self):
        return self.model.lstm_decoder

    @property
    def lstm_encoder(self):
        return self.model.lstm_encoder

    def forward(self, imgs: Tensor, targets: Optional[Tensor] = None):
        return self.model(imgs, targets)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        logits, loss = self.model.forward_teacher_forcing(imgs, targets)
        self.log("train_loss", loss, sync_dist=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.val_or_test_step(batch)

    def test_step(self, batch, batch_idx):
        return self.val_or_test_step(batch)

    def val_or_test_step(self, batch) -> Tensor:
        imgs, targets = batch
        logits, _, loss = self(imgs, targets)
        _, preds = logits.max(-1)

        # Update and log metrics.
        self.model.cer_metric(preds, targets)
        self.model.wer_metric(preds, targets)
        self.log("char_error_rate", self.model.cer_metric, prog_bar=True)
        self.log("word_error_rate", self.model.wer_metric, prog_bar=True)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        def _decay_factor(
            epoch: int, max_expo: int, factor: float, decay_every: int = 10000
        ):
            # Exponential decay.
            expo = int(self.global_step / decay_every)
            if expo >= max_expo:
                return factor ** max_expo
            return factor ** expo

        factor = 0.9
        min_lr = 1e-5

        if self.use_lr_scheduler:
            # max_expo is calculated as n where `lr * factor^n = 1e-5`
            max_expo = math.floor(
                math.log(min_lr / self.learning_rate) / math.log(factor)
            )
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                partial(_decay_factor, max_expo=max_expo, factor=factor),
                verbose=True,
            )
            return [optimizer], [scheduler]
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        # fmt: off
        parser = parent_parser.add_argument_group("ShowAttendRead")
        parser.add_argument("--sar_learning_rate", type=float, default=0.001)
        parser.add_argument("--sar_d_enc", type=int, default=512)
        # parser.add_argument("--sar_d_model", type=int, default=512)
        parser.add_argument("--sar_d_k", type=int, default=512)
        parser.add_argument("--sar_resnet", type=str, default="resnet18",
                            choices=["resnet18", "resnet34", "resnet50", "resnet31"],
                            help="Image encoder to use. Resnet{18,34,50} are "
                                 "standard ResNet architectures (11.3M, "
                                 "21.4M, 24.0M parameters respectively); Resnet31 is a "
                                 "modified ResNet for HTR (46.0M parameters).")
        parser.add_argument("--use_lr_scheduler", action="store_true",
                            help="Use a exponentially decaying learning rate.")
        parser.add_argument("--sar_dec_dropout", type=float, default=0.0,
                            help="Decoder dropout.")
        parser.add_argument("--sar_enc_dropout", type=float, default=0.1,
                            help="Encoder dropout.")
        parser.add_argument("--sar_pred_dropout", type=float, default=0.1,
                            help="Prediction dropout.")
        return parent_parser
        # fmt: on


class LitFullPageHTREncoderDecoder(pl.LightningModule):
    model: FullPageHTREncoderDecoder

    """
    Pytorch Lightning module that acting as a wrapper around the
    FullPageHTREncoderDecoder class.

    Using a PL module allows the model to be used in conjunction with a Pytorch
    Lightning Trainer, and takes care of logging relevant metrics to Tensorboard.
    """

    def __init__(
        self,
        label_encoder: LabelEncoder,
        learning_rate: float = 0.0002,
        max_seq_len: int = 500,
        d_model: int = 260,
        num_layers: int = 6,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        encoder_name: str = "resnet18",
        drop_enc: int = 0.1,
        drop_dec: int = 0.1,
        activ_dec: str = "gelu",
        loss_reduction: str = "mean",
        vocab_len: Optional[int] = None,  # if not specified len(label_encoder) is used
        params_to_log: Optional[Dict[str, Union[str, float, int]]] = None,
    ):
        super().__init__()

        # Save hyperparameters.
        self.learning_rate = learning_rate
        if params_to_log is not None:
            self.save_hyperparameters(params_to_log)
        self.save_hyperparameters(
            "learning_rate",
            "d_model",
            "num_layers",
            "nhead",
            "dim_feedforward",
            "max_seq_len",
            "encoder_name",
            "drop_enc",
            "drop_dec",
            "activ_dec",
        )

        # Initialize the model.
        self.model = FullPageHTREncoderDecoder(
            label_encoder=label_encoder,
            max_seq_len=max_seq_len,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            encoder_name=encoder_name,
            drop_enc=drop_enc,
            drop_dec=drop_dec,
            activ_dec=activ_dec,
            vocab_len=vocab_len,
            loss_reduction=loss_reduction,
        )

    @property
    def encoder(self):
        return self.model.encoder

    @property
    def decoder(self):
        return self.model.decoder

    def forward(self, imgs: Tensor, targets: Optional[Tensor] = None):
        return self.model(imgs, targets)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        logits, loss = self.model.forward_teacher_forcing(imgs, targets)
        self.log("train_loss", loss, sync_dist=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.val_or_test_step(batch)

    def test_step(self, batch, batch_idx):
        return self.val_or_test_step(batch)

    def val_or_test_step(self, batch) -> Tensor:
        imgs, targets = batch
        logits, _, loss = self(imgs, targets)
        _, preds = logits.max(-1)

        # Update and log metrics.
        self.model.cer_metric(preds, targets)
        self.model.wer_metric(preds, targets)
        self.log("char_error_rate", self.model.cer_metric, prog_bar=True)
        self.log("word_error_rate", self.model.wer_metric, prog_bar=True)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # fmt: off
        parser = parent_parser.add_argument_group("FullPageHTREncoderDecoder")
        parser.add_argument("--fphtr_learning_rate", type=float, default=0.0002)
        parser.add_argument("--fphtr_encoder", type=str, default="resnet18",
                            choices=["resnet18", "resnet34", "resnet50", "resnet31"],
                            help="Image encoder to use. Resnet{18,34,50} are "
                                 "standard ResNet architectures (11.3M, "
                                 "21.4M, 24.0M parameters respectively); Resnet31 is a "
                                 "modified ResNet for HTR (46.0M parameters).")
        parser.add_argument("--fphtr_d_model", type=int, default=260)
        parser.add_argument("--fphtr_num_layers", type=int, default=6)
        parser.add_argument("--fphtr_nhead", type=int, default=4)
        parser.add_argument("--fphtr_dim_feedforward", type=int, default=1024)
        parser.add_argument("--fphtr_drop_enc", type=float, default=0.1,
                            help="Encoder dropout.")
        parser.add_argument("--fphtr_drop_dec", type=float, default=0.1,
                            help="Decoder dropout.")
        return parent_parser
        # fmt: on
