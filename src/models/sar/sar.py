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

from src.models.sar.resnet31_htr import ResNet31HTR
from src.metrics import CharacterErrorRate, WordErrorRate
from src.util import LabelEncoder


class SARDecoder(nn.Module):
    """Implementation Decoder module in `SAR.

    <https://arxiv.org/abs/1811.00751>`_.

    Args:
        num_classes (int): Output class number :math:`C`.
        d_model (int): Dim of channels from backbone :math:`D_i`.
        d_enc (int): Dim of encoder RNN layer :math:`D_m`.
        d_k (int): Dim of channels of attention module.
        dec_dropout (float): Dropout of RNN layer in decoder.
        pred_dropout (float): Dropout probability of prediction layer.
        max_seq_len (int): Maximum sequence length for decoding.
        enc_bi_rnn (bool): If True, use bidirectional RNN in encoder.
        dec_bi_rnn (bool): If True, use bidirectional RNN in decoder.
        dec_gru (bool): If True, use GRU, else LSTM in decoder.
        mask (bool): If True, mask padding in feature map.
        sos_tkn_idx (int): Index of start token.
        pad_tkn_idx (int): Index of padding token.
        pred_concat (bool): If True, concat glimpse feature from
            attention with holistic feature and hidden state.

    Warning:
        This decoder will not predict the final class which is assumed to be
        `<PAD>`. Therefore, its output size is always :math:`C - 1`. `<PAD>`
        is also ignored by loss as specified in
        :obj:`mmocr.models.textrecog.recognizer.EncodeDecodeRecognizer`.
    """

    def __init__(
        self,
        num_classes,
        pad_tkn_idx,
        sos_tkn_idx,
        eos_tkn_idx,
        d_model=512,
        d_enc=512,
        d_k=64,
        dec_dropout=0.0,
        pred_dropout=0.0,
        max_seq_len=50,
        dec_gru=False,
        enc_bi_rnn=False,
        dec_bi_rnn=False,
        mask=True,
        pred_concat=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.eos_tkn_idx = eos_tkn_idx
        self.sos_tkn_idx = sos_tkn_idx
        self.pad_tkn_idx = pad_tkn_idx
        self.enc_bi_rnn = enc_bi_rnn
        self.d_k = d_k
        self.sos_tkn_idx = sos_tkn_idx
        self.max_seq_len = max_seq_len
        self.mask = mask
        self.pred_concat = pred_concat

        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        decoder_rnn_out_size = encoder_rnn_out_size * (int(dec_bi_rnn) + 1)
        # 2D attention layer
        self.conv1x1_1 = nn.Linear(decoder_rnn_out_size, d_k)
        self.conv3x3_1 = nn.Conv2d(d_model, d_k, kernel_size=3, stride=1, padding=1)
        self.conv1x1_2 = nn.Linear(d_k, 1)

        # Decoder RNN layer
        kwargs = dict(
            input_size=encoder_rnn_out_size,
            hidden_size=encoder_rnn_out_size,
            num_layers=2,
            batch_first=True,
            dropout=dec_dropout,
            bidirectional=dec_bi_rnn,
        )
        if dec_gru:
            self.rnn_decoder = nn.GRU(**kwargs)
        else:
            self.rnn_decoder = nn.LSTM(**kwargs)

        # Decoder input embedding
        self.embedding = nn.Embedding(
            self.num_classes, encoder_rnn_out_size, padding_idx=pad_tkn_idx
        )

        # Prediction layer
        self.pred_dropout = nn.Dropout(pred_dropout)
        if pred_concat:
            fc_in_channel = decoder_rnn_out_size + d_model + encoder_rnn_out_size
        else:
            fc_in_channel = d_model
        self.prediction = nn.Linear(fc_in_channel, num_classes)

    def _2d_attention(self, decoder_input, feat, holistic_feat, valid_ratios=None):
        y = self.rnn_decoder(decoder_input)[0]  # bsz * (seq_len + 1) * hidden_size

        attn_query = self.conv1x1_1(y)  # bsz * (seq_len + 1) * attn_size
        bsz, seq_len, attn_size = attn_query.size()
        attn_query = attn_query.view(bsz, seq_len, attn_size, 1, 1)

        attn_key = self.conv3x3_1(feat)  # bsz * attn_size * h * w
        attn_key = attn_key.unsqueeze(1)  # bsz * 1 * attn_size * h * w

        # bsz * (seq_len + 1) * attn_size * h * w
        attn_weight = torch.tanh(torch.add(attn_key, attn_query, alpha=1))
        # bsz * (seq_len + 1) * h * w * attn_size
        attn_weight = attn_weight.permute(0, 1, 3, 4, 2).contiguous()
        attn_weight = self.conv1x1_2(attn_weight)  # bsz * (seq_len + 1) * h * w * 1
        bsz, T, h, w, c = attn_weight.size()
        assert c == 1

        if valid_ratios is not None:
            # cal mask of attention weight
            attn_mask = torch.zeros_like(attn_weight)
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(w, math.ceil(w * valid_ratio))
                attn_mask[i, :, :, valid_width:, :] = 1
            attn_weight = attn_weight.masked_fill(attn_mask.bool(), float("-inf"))

        attn_weight = attn_weight.view(bsz, T, -1)
        attn_weight = F.softmax(attn_weight, dim=-1)
        # bsz * (seq_len + 1) * 1 * h * w
        attn_weight = (
            attn_weight.view(bsz, T, h, w, c).permute(0, 1, 4, 2, 3).contiguous()
        )

        # bsz * (seq_len + 1) * C
        attn_feat = torch.sum(
            torch.mul(feat.unsqueeze(1), attn_weight), (3, 4), keepdim=False
        )

        # linear transformation
        if self.pred_concat:
            hf_c = holistic_feat.size(-1)
            holistic_feat = holistic_feat.expand(bsz, seq_len, hf_c)
            y = self.prediction(torch.cat((y, attn_feat, holistic_feat), 2))
        else:
            y = self.prediction(attn_feat)
        y = self.pred_dropout(y)

        return y  # bsz * (seq_len + 1) * num_classes

    def forward_teacher_forcing(self, feat, out_enc, targets, img_metas=None):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape :math:`(N, D_m)`.
            targets (Tensor): A Tensor of shape :math:`(N, T)`. Each element
                is the index of a character.
            img_metas (dict): A dict that contains meta information of input
                images. Preferably with the key ``valid_ratio``.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C-1)`.
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

        tgt_embedding = self.embedding(targets)  # bsz * seq_len * emb_dim
        out_enc = out_enc.unsqueeze(1)  # bsz * 1 * emb_dim
        in_dec = torch.cat((out_enc, tgt_embedding), dim=1)  # bsz * (seq_len + 1) * C
        # bsz * (seq_len + 1) * num_classes
        logits = self._2d_attention(in_dec, feat, out_enc, valid_ratios=valid_ratios)

        return logits[:, 1:, :]  # bsz * seq_len * num_classes

    def forward(self, feat, out_enc, img_metas=None):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            img_metas (dict): A dict that contains meta information of input
                images. Preferably with the key ``valid_ratio``.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C-1)`.
            Tensor: A tensor of sampled token ids of shape :math:`(N, T, C-1)`.
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

        seq_len = self.max_seq_len

        bsz = feat.size(0)
        start_token = torch.full(
            (bsz,), self.sos_tkn_idx, device=feat.device, dtype=torch.long
        )
        sampled_ids = [start_token]
        start_token = self.embedding(start_token)  # bsz * emb_dim
        # bsz * seq_len * emb_dim
        start_token = start_token.unsqueeze(1).expand(-1, seq_len, -1)
        out_enc = out_enc.unsqueeze(1)  # bsz * 1 * emb_dim
        # bsz * (seq_len + 1) * emb_dim
        decoder_input = torch.cat((out_enc, start_token), dim=1)

        logits = []
        for i in range(1, seq_len + 1):
            decoder_output = self._2d_attention(
                decoder_input, feat, out_enc, valid_ratios=valid_ratios
            )
            char_output = decoder_output[:, i, :]  # bsz * num_classes
            _, pred = torch.max(char_output, dim=1, keepdim=False)
            logits.append(char_output)
            sampled_ids.append(pred)
            char_embedding = self.embedding(pred)  # bsz * emb_dim
            if i < seq_len:
                decoder_input[:, i + 1, :] = char_embedding

        logits = torch.stack(logits, 1)  # bsz * seq_len * num_classes
        sampled_ids = torch.stack(sampled_ids, 1)

        # Replace all tokens in `sampled_ids` after <EOS> with <PAD> tokens.
        eos_idxs = (sampled_ids == self.eos_tkn_idx).float().argmax(1)
        for i in range(bsz):
            if eos_idxs[i] != 0:  # sampled sequence contains <EOS> token
                sampled_ids[i, eos_idxs[i] + 1 :] = self.pad_tkn_idx

        return logits, sampled_ids


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
