"""
Implementation of Show Attend and Read (SAR),
modified from https://github.com/open-mmlab/mmocr
"""

import math
from typing import Tuple, Optional, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.sar.sar_decoder import SARDecoder
from src.models.sar.resnet31_htr import ResNet31HTR
from src.metrics import CharacterErrorRate, WordErrorRate
from src.util import LabelEncoder


class SAREncoder(nn.Module):
    """Implementation of encoder module in `SAR.

    <https://arxiv.org/abs/1811.00751>`_.

    Args:
        bidirectional (bool): If True, use bidirectional RNN in encoder.
        dropout (float): Dropout probability of RNN layer in encoder.
        enc_gru (bool): If True, use GRU, else LSTM in encoder.
        d_model (int): Dim :math:`D_i` of channels from backbone.
        d_enc (int): Dim :math:`D_m` of encoder RNN layer.
        mask (bool): If True, mask padding in RNN sequence.
    """

    def __init__(
        self,
        bidirectional=False,
        dropout=0.0,
        enc_gru=False,
        d_model=512,
        d_enc=512,
        mask=True,
    ):
        super().__init__()
        assert isinstance(bidirectional, bool)
        assert isinstance(dropout, (int, float))
        assert 0 <= dropout < 1.0
        assert isinstance(enc_gru, bool)
        assert isinstance(d_model, int)
        assert isinstance(d_enc, int)
        assert isinstance(mask, bool)

        self.bidirectional = bidirectional
        self.dropout = dropout
        self.mask = mask

        # LSTM Encoder
        kwargs = dict(
            input_size=d_model,
            hidden_size=d_enc,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        if enc_gru:
            self.rnn_encoder = nn.GRU(**kwargs)
        else:
            self.rnn_encoder = nn.LSTM(**kwargs)

        # global feature transformation
        encoder_rnn_out_size = d_enc * (int(bidirectional) + 1)
        self.linear = nn.Linear(encoder_rnn_out_size, encoder_rnn_out_size)

    def forward(self, feat, img_metas=None):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            img_metas (dict): A dict that contains meta information of input
                images. Preferably with the key ``valid_ratio``.

        Returns:
            Tensor: A tensor of shape :math:`(N, D_m)`.
        """
        if img_metas is not None:
            assert isinstance(img_metas, list)
            assert all(isinstance(item, dict) for item in img_metas)
            assert len(img_metas) == feat.size(0)

        valid_ratios = None
        if img_metas is not None:
            valid_ratios = (
                [img_meta.get("valid_ratio", 1.0) for img_meta in img_metas]
                if self.mask
                else None
            )

        h_feat = feat.size(2)
        feat_v = F.max_pool2d(feat, kernel_size=(h_feat, 1), stride=1, padding=0)
        feat_v = feat_v.squeeze(2)  # bsz * C * W
        feat_v = feat_v.permute(0, 2, 1).contiguous()  # bsz * W * C

        holistic_feat = self.rnn_encoder(feat_v)[0]  # bsz * T * C

        if valid_ratios is not None:
            valid_hf = []
            T = holistic_feat.size(1)
            for i, valid_ratio in enumerate(valid_ratios):
                valid_step = min(T, math.ceil(T * valid_ratio)) - 1
                valid_hf.append(holistic_feat[i, valid_step, :])
            valid_hf = torch.stack(valid_hf, dim=0)
        else:
            valid_hf = holistic_feat[:, -1, :]  # bsz * C

        holistic_feat = self.linear(valid_hf)  # bsz * C

        return holistic_feat


class ShowAttendRead(nn.Module):
    """See https://arxiv.org/abs/1811.00751"""

    resnet_encoder: ResNet31HTR
    lstm_encoder: SAREncoder
    lstm_decoder: SARDecoder
    cer_metric: CharacterErrorRate
    wer_metric: WordErrorRate
    loss_fn: Callable
    label_encoder: LabelEncoder

    def __init__(
        self,
        label_encoder: LabelEncoder,
        max_seq_len: int = 50,
        d_enc: int = 512,
        d_model: int = 512,
        d_k: int = 512,
        dec_dropout: int = 0.0,
        enc_dropout: int = 0.1,
        pred_dropout: int = 0.1,
        loss_reduction: str = "mean",
        vocab_len: Optional[int] = None,
    ):
        super().__init__()

        self.label_encoder = label_encoder

        # Obtain special token indices.
        self.eos_tkn_idx, self.sos_tkn_idx, self.pad_tkn_idx = label_encoder.transform(
            ["<EOS>", "<SOS>", "<PAD>"]
        )

        self.resnet_encoder = ResNet31HTR(
            base_channels=1,
            layers=[1, 2, 5, 3],
            channels=[64, 128, 256, 256, 512, 512, d_model],
        )
        self.lstm_encoder = SAREncoder(
            dropout=enc_dropout,
            d_model=d_model,
            d_enc=d_enc,
            bidirectional=False,
            enc_gru=False,
        )
        self.lstm_decoder = SARDecoder(
            num_classes=(vocab_len or label_encoder.n_classes),
            pad_tkn_idx=self.pad_tkn_idx,
            sos_tkn_idx=self.sos_tkn_idx,
            eos_tkn_idx=self.eos_tkn_idx,
            d_model=d_model,
            d_enc=d_enc,
            d_k=d_k,
            dec_dropout=dec_dropout,
            pred_dropout=pred_dropout,
            max_seq_len=max_seq_len,
            dec_gru=False,
            enc_bi_rnn=False,
            dec_bi_rnn=False,
            pred_concat=True,
        )

        # Initialize metrics and loss function.
        self.cer_metric = CharacterErrorRate(label_encoder)
        self.wer_metric = WordErrorRate(label_encoder)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.pad_tkn_idx, reduction=loss_reduction
        )

    def forward_teacher_forcing(
        self, imgs: Tensor, targets: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Run inference on the model using greedy decoding and teacher forcing.

        Teacher forcing implies that at each decoding time step, the ground truth
        target of the previous time step is fed as input to the model.

        Returns:
            - logits, obtained at each time step during decoding
            - loss value
        """
        feats = self.resnet_encoder(imgs)
        h_holistic = self.lstm_encoder(feats)
        logits = self.lstm_decoder.forward_teacher_forcing(feats, h_holistic, targets)
        loss = self.loss_fn(logits.transpose(1, 2), targets)
        return logits, loss

    def forward(
        self, imgs: Tensor, targets: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Union[Tensor, None]]:
        """
        Run inference on the model using greedy decoding.

        Returns:
            - logits, obtained at each time step during decoding
            - sampled class indices, i.e. model predictions, obtained by applying
                  greedy decoding (argmax on logits) at each time step
            - loss value (only calculated when specifiying `targets`, otherwise
                  defaults to None)
        """
        feats = self.resnet_encoder(imgs)
        h_holistic = self.lstm_encoder(feats)
        logits, sampled_ids = self.lstm_decoder(feats, h_holistic, targets)

        loss = None
        if targets is not None:
            loss = self.loss_fn(
                logits[:, : targets.size(1), :].transpose(1, 2),
                targets[:, : logits.size(1)],
            )
        return logits, sampled_ids, loss
