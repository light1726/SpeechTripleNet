import torch
import torch.nn as nn
import torch.distributions as D

from .utils import get_activation, DownSampleEncBlk, VectorQuantizerEMA


class TIVConvEncoder(nn.Module):
    """
    Encode time-invariant features, e.g., speaker identity, emotion
    """

    def __init__(self, in_channels, h_channels, out_channels, conv_kernels,
                 paddings, pooling_type, pooling_kernels, activation):
        super(TIVConvEncoder, self).__init__()
        self.out_channels = out_channels
        assert len(conv_kernels) == len(paddings) and len(paddings) == len(pooling_kernels)
        self.activation = get_activation(activation)
        self.conv_initial = nn.Conv1d(in_channels, h_channels, kernel_size=(1,), padding=0)
        self.conv_layers = nn.ModuleList([DownSampleEncBlk(
            h_channels, h_channels, kernel_size=k, padding=p,
            activation=self.activation, pooling_kernel=pk, pooling_type=pooling_type)
            for k, p, pk in zip(conv_kernels, paddings, pooling_kernels)])
        self.out_layer_norm = nn.LayerNorm(h_channels)
        self.mu_logs_linear = nn.Linear(h_channels, self.out_channels * 2)

    def forward(self, x):
        """
        :param x: [batch, channels, time-length]
        :return: mu and logs of shape [batch, out_channels]
        """
        h = self.conv_initial(x)
        for conv in self.conv_layers:
            h = conv(h)
        h = self.activation(h)
        h = self.out_layer_norm(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = torch.mean(h, dim=[2])  # [batch, h_channels]
        mu_logs = self.mu_logs_linear(h)
        mu, logs = torch.split(mu_logs, self.out_channels, dim=1)
        return mu, logs

    @staticmethod
    def logp_z_given_x(z, mu, logs):
        """
        :param z: [batch, out_chanvnels]
        :param mu: [batch, out_channels]
        :param logs: [batch, out_channels]
        :return: [batch]
        """
        dist = D.Normal(mu, logs.exp())
        logprobs = dist.log_prob(z)
        logprobs = torch.sum(logprobs, dim=1)
        return logprobs

    @staticmethod
    def sample(mu, logs, std=1.):
        """
        :param mu: [batch, channel]
        :param logs:[batch, channel]
        :param std: sampling std
        :return: sample: [batch, channel]
        """
        assert std > 0.
        eps = torch.normal(torch.zeros_like(mu), torch.ones_like(logs) * std)
        sample = eps * torch.exp(logs) + mu
        return sample

    @staticmethod
    def kl_divergence(mu, log_std):
        """
        :param mu: [B, C]
        :param log_std: [B, C]
        :return:
        """
        post = D.Normal(mu, torch.exp(log_std))
        prior = D.Normal(
            torch.zeros_like(mu, requires_grad=False),
            torch.ones_like(log_std, requires_grad=False))
        kl = D.kl.kl_divergence(post, prior)
        kl = torch.mean(torch.sum(kl, dim=1))
        return kl


class VQConvEncoder(nn.Module):
    def __init__(self, in_channels, h_channels, codebook_size, beta, decay, out_channels,
                 conv_kernels, paddings, pooling_kernels, activation, device):
        super(VQConvEncoder, self).__init__()
        self.device = device
        assert len(conv_kernels) == len(paddings) and len(paddings) == len(pooling_kernels)
        self.activation = get_activation(activation)
        self.conv_initial = nn.Conv1d(in_channels, h_channels, kernel_size=1, padding=0)
        self.conv_layers = nn.ModuleList(
            [DownSampleEncBlk(h_channels, h_channels, kernel_size=k, padding=p,
                              activation=self.activation, pooling_kernel=pk)
             for k, p, pk in zip(conv_kernels, paddings, pooling_kernels)])
        self.out_linear = nn.Linear(h_channels, out_channels)
        self.out_layer_norm = nn.LayerNorm(out_channels)
        torch.nn.init.zeros_(self.out_linear.weight)
        torch.nn.init.zeros_(self.out_linear.bias)
        self.vq_layer = VectorQuantizerEMA(device, codebook_size, out_channels, beta, decay)

    def forward(self, x):
        """
        :param x: [B, C, T]
        :return: [B, C]
        """
        h = self.conv_initial(x)
        for conv in self.conv_layers:
            h = conv(h)
        h = self.activation(h)
        h_avg = torch.mean(h, dim=[2])  # [batch, h_channels]
        z_e = self.out_linear(h_avg)
        z_e = self.out_layer_norm(z_e)
        loss, z_q, perplexity, min_encodings = self.vq_layer(z_e)
        return z_e, z_q, loss, perplexity

    @staticmethod
    def kl_divergence(mu, log_std):
        """
        :param mu: [B, C]
        :param log_std: [B, C]
        :return:
        """
        post = D.Normal(mu, torch.exp(log_std))
        prior = D.Normal(
            torch.zeros_like(mu, requires_grad=False),
            torch.ones_like(log_std, requires_grad=False))
        kl = D.kl.kl_divergence(post, prior)
        kl = torch.mean(torch.sum(kl, dim=1))
        return kl
