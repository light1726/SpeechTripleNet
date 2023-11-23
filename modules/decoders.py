import torch
import torch.nn as nn

from .utils import ConvResLNBlk, ConvResBNBlk, MaskedBatchNorm1d, get_activation, sequence_mask
from transformer import get_sinusoid_encoding_table, FFTBlock


class ConvDecoder(nn.Module):
    def __init__(self, in_channels, h_channels, out_channels,
                 conv_kernels, paddings, activation):
        super(ConvDecoder, self).__init__()
        self.out_channels = out_channels
        self.activation = get_activation(activation)
        self.conv_initial = nn.Conv1d(in_channels, h_channels, kernel_size=1, padding=0)
        self.conv_layers = nn.ModuleList([
            ConvResLNBlk(h_channels, h_channels, kernel_size=k,
                         padding=p, activation=self.activation)
            for k, p in zip(conv_kernels, paddings)])
        self.out_layer_norm = nn.LayerNorm(h_channels)
        self.out_linear = nn.Conv1d(h_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        """
        :param x: [B, C_in, T]
        :return: output: [B, C_out, T]
        """
        h = self.conv_initial(x)
        for conv in self.conv_layers:
            h = conv(h)
        h = self.activation(h)
        h = self.out_layer_norm(h.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.out_linear(h)
        return output


class TransformerDecoder(nn.Module):
    """ TransformerDecoder """

    def __init__(self, decoder_hidden, decoder_layer, decoder_head, conv_filter_size,
                 conv_kernel_size, decoder_dropout, out_channels, max_seq_len):
        super(TransformerDecoder, self).__init__()

        n_position = max_seq_len + 1
        d_k = d_v = decoder_hidden // decoder_head
        self.max_seq_len = max_seq_len
        self.d_model = decoder_hidden

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, decoder_hidden).unsqueeze(0),
            requires_grad=False)

        self.layer_stack = nn.ModuleList(
            [FFTBlock(
                decoder_hidden, decoder_head, d_k, d_v, conv_filter_size,
                conv_kernel_size, dropout=decoder_dropout)
                for _ in range(decoder_layer)])
        self.out_linear = nn.Linear(decoder_hidden, out_channels)

    def forward(self, enc_seq, mask):
        enc_seq = enc_seq.permute(0, 2, 1)  # [B, T, C]
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                                                   :, :max_len, :
                                                   ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
        outputs = self.out_linear(dec_output).permute(0, 2, 1)
        return outputs


class ConvSADecoder(nn.Module):
    def __init__(self, in_channels, h_channels, conv_kernels, paddings,
                 activation, sa_hidden, sa_layer, sa_head, sa_filter_size,
                 sa_kernel_size, sa_dropout, out_channels, max_seq_len):
        super(ConvSADecoder, self).__init__()
        # convolutions
        self.out_channels = out_channels
        self.activation = get_activation(activation)
        self.conv_initial = nn.Conv1d(in_channels, h_channels, kernel_size=1, padding=0)
        self.conv_layers = nn.ModuleList([
            ConvResLNBlk(h_channels, h_channels, kernel_size=k,
                         padding=p, activation=self.activation)
            for k, p in zip(conv_kernels, paddings)])
        self.out_layer_norm = nn.LayerNorm(h_channels)

        # self-attentions
        n_position = max_seq_len + 1
        d_k = d_v = sa_hidden // sa_head
        self.max_seq_len = max_seq_len
        self.d_model = sa_hidden

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, sa_hidden).unsqueeze(0),
            requires_grad=False)

        self.layer_stack = nn.ModuleList(
            [FFTBlock(sa_hidden, sa_head, d_k, d_v, sa_filter_size,
                      sa_kernel_size, dropout=sa_dropout)
             for _ in range(sa_layer)])

        self.out_linear = nn.Conv1d(h_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, lens=None):
        """
        :param x: [B, C, T]
        :param lens: [B]
        :return: mu and log_std of shape [B, C, T]
        """
        if lens is not None:
            max_len = x.size(2)
            mask = ~sequence_mask(lens, max_length=max_len)
        else:
            B, T = x.size(0), x.size(2)
            mask = torch.zeros([B, T], dtype=torch.bool, device=x.device)
        # convolution
        h = self.conv_initial(x)
        for conv in self.conv_layers:
            h = conv(h)
        h = self.activation(h)
        h = self.out_layer_norm(h.permute(0, 2, 1)).permute(0, 2, 1)
        # self-attention
        sa_input = h.permute(0, 2, 1)  # [B, T, C]
        batch_size, max_len = sa_input.shape[0], sa_input.shape[1]

        # -- Forward
        if not self.training and sa_input.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = sa_input + get_sinusoid_encoding_table(
                sa_input.shape[1], self.d_model
            )[: sa_input.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                sa_input.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = sa_input[:, :max_len, :] + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask)

        dec_output = dec_output.permute(0, 2, 1).contiguous()
        output = self.out_linear(dec_output)
        return output


class ConvSABNDecoder(nn.Module):
    def __init__(self, in_channels, h_channels, conv_kernels, paddings,
                 activation, sa_hidden, sa_layer, sa_head, sa_filter_size,
                 sa_kernel_size, sa_dropout, out_channels, max_seq_len):
        super(ConvSABNDecoder, self).__init__()
        # convolutions
        self.out_channels = out_channels
        self.activation = get_activation(activation)
        self.conv_initial = nn.Conv1d(in_channels, h_channels, kernel_size=1, padding=0)
        self.conv_layers = nn.ModuleList([
            ConvResBNBlk(h_channels, h_channels, kernel_size=k,
                         padding=p, activation=self.activation)
            for k, p in zip(conv_kernels, paddings)])
        self.out_batch_norm = MaskedBatchNorm1d(h_channels)

        # self-attentions
        n_position = max_seq_len + 1
        d_k = d_v = sa_hidden // sa_head
        self.max_seq_len = max_seq_len
        self.d_model = sa_hidden

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, sa_hidden).unsqueeze(0),
            requires_grad=False)

        self.layer_stack = nn.ModuleList(
            [FFTBlock(sa_hidden, sa_head, d_k, d_v, sa_filter_size,
                      sa_kernel_size, dropout=sa_dropout)
             for _ in range(sa_layer)])

        self.out_linear = nn.Conv1d(h_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, lens=None):
        """
        :param x: [B, C, T]
        :param lens: [B]
        :return: mu and log_std of shape [B, C, T]
        """
        if lens is not None:
            max_len = x.size(2)
            mask = ~sequence_mask(lens, max_length=max_len)
        else:
            B, T = x.size(0), x.size(2)
            mask = torch.zeros([B, T], dtype=torch.bool, device=x.device)
        # convolution
        h = self.conv_initial(x)
        for conv in self.conv_layers:
            h = conv(h, lens)
        h = self.activation(h)
        h = self.out_batch_norm(h, lens)
        # self-attention
        sa_input = h.permute(0, 2, 1)  # [B, T, C]
        batch_size, max_len = sa_input.shape[0], sa_input.shape[1]

        # -- Forward
        if not self.training and sa_input.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = sa_input + get_sinusoid_encoding_table(
                sa_input.shape[1], self.d_model
            )[: sa_input.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                sa_input.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = sa_input[:, :max_len, :] + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask)

        dec_output = dec_output.permute(0, 2, 1).contiguous()
        output = self.out_linear(dec_output)
        return output
