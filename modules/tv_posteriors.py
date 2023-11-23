import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from transformer import get_sinusoid_encoding_table, FFTBlock
from .utils import sequence_mask, ConvResLNBlk, ConvResBNBlk,\
    get_activation, ConvINBlk, ConvBNBlk, masked_instance_norm, MaskedBatchNorm1d


class BaseTVContEncoder(nn.Module):
    """
    Time-variant feature encoder
    """

    def __init__(self, device, **kwargs):
        super(BaseTVContEncoder, self).__init__()

    def forward(self, x):
        """
        :param x: [B, C, T]
        :return: mu: [B, C, T], log_std: [B, C, T]
        """
        pass

    @staticmethod
    def sample(mu, log_std, std=1.):
        """
        :param mu: [B, C, T]
        :param log_std: [B, C, T]
        :param std: sampling std, positive float
        :return: z: [B, C, T]
        """
        assert std > 0.
        eps = torch.normal(torch.zeros_like(mu), torch.ones_like(log_std) * std)
        z = eps * torch.exp(log_std) + mu
        return z

    def logp_z_given_x(self, z, mu, log_std, lens=None):
        """
        :param z: [B, C, T]
        :param mu: [B, C, T]
        :param log_std: [B, C, T]
        :param lens: [B]
        :return: [batch]
        """
        dist = D.Normal(mu, log_std.exp())
        if lens is not None:
            mask = sequence_mask(lens, max_length=z.size(2)).to(dtype=z.dtype)
        else:
            B = z.size(0)
            T = z.size(2)
            mask = torch.ones([B, T], dtype=z.dtype, device=self.device)
            lens = torch.ones([B]) * T
        logprobs = dist.log_prob(z)
        logprobs = torch.sum(torch.sum(logprobs, dim=1) * mask, dim=1) / lens.to(z.dtype)  # [B,]
        return logprobs

    @staticmethod
    def kl_divergence(mu, log_std, lens=None):
        """
        :param mu: [B, C, T]
        :param log_std: [B, C, T]
        :param lens: [B]
        :return:
        """
        if lens is not None:
            mask = sequence_mask(lens).to(mu.dtype)
        else:
            B = mu.size(0)
            T = mu.size(2)
            mask = torch.ones([B, T], dtype=mu.dtype, device=mu.device)
        post = D.Normal(mu, torch.exp(log_std))
        prior = D.Normal(
            torch.zeros_like(mu, requires_grad=False),
            torch.ones_like(log_std, requires_grad=False))
        kl = D.kl.kl_divergence(post, prior)
        kl = torch.sum(torch.sum(kl, dim=1) * mask, dim=1) / torch.sum(mask, dim=1)
        kl = torch.mean(kl)
        return kl


class BaseTVDiscEncoder(nn.Module):
    """
    Time-variant feature encoder
    """

    def __init__(self, device, n_class, temperature, **kwargs):
        super(BaseTVDiscEncoder, self).__init__()
        self.device = device
        self.disc_dim = n_class
        self.temperature = temperature

    def forward(self, x):
        """
        :param x: [B, C, T]
        :return: mu: [B, C, T], log_std: [B, C, T]
        """
        pass

    def sample(self, logits):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.
        Parameters
        ----------
        logits : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D, T)
        """
        logits = logits.permute(0, 2, 1)  # [B, T, D]
        samples = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        return samples

    def embed(self, idx):
        """
        :param idx: [B, T]
        :return:
        """
        onehot = F.one_hot(idx, num_classes=self.disc_dim)
        return torch.matmul(onehot.float(), self.embeds).permute(0, 2, 1)

    def kl_divergence(self, logits, lens, EPS=1e-12):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.
        Parameters
        ----------
        logits : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D, T)
        """
        alpha = F.softmax(logits, dim=1)
        B, D, T = alpha.size()
        if lens is not None:
            mask = sequence_mask(lens).to(alpha.dtype)
        else:
            mask = torch.ones([B, T], dtype=alpha.dtype, device=alpha.device)
        alpha = alpha.permute(0, 2, 1).reshape([B * T, D])
        log_dim = torch.log(torch.ones([1], dtype=alpha.dtype, device=self.device) * self.disc_dim)
        # Calculate negative entropy of each row
        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
        # Take mean of negative entropy across batch
        neg_entropy = neg_entropy.reshape([B, T])
        mean_neg_entropy = torch.mean(
            torch.sum(neg_entropy * mask, dim=1) / torch.sum(mask, dim=1))
        # KL loss of alpha with uniform categorical variable
        kl_loss = log_dim + mean_neg_entropy
        return kl_loss

    @staticmethod
    def kl_divergence_with_prior(logits, prior, lens, EPS=1e-12):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.
        Parameters
        ----------
        logits : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D, T)
        prior : torch.Tensor
            shape (N, T)
        """
        prior = F.one_hot(prior)  # [B, T, D]
        alpha = F.softmax(logits, dim=1)
        B, D, T = alpha.size()
        if lens is not None:
            mask = sequence_mask(lens).to(alpha.dtype)
        else:
            mask = torch.ones([B, T], dtype=alpha.dtype, device=alpha.device)
        alpha = alpha.permute(0, 2, 1).reshape([B * T, D])
        prior = prior.reshape([B * T, D])
        kl_loss = torch.sum(alpha * (torch.log(alpha + EPS) - torch.log(prior + EPS)), dim=1)  # [B * T]
        # kl_loss = torch.sum(prior * (torch.log(prior + EPS) - torch.log(alpha + EPS)), dim=1)
        kl_loss = kl_loss.reshape([B, T])
        kl_loss = torch.mean(torch.sum(kl_loss * mask, dim=1) / torch.sum(mask, dim=1))
        return kl_loss


class TVConvSAINEncoder(BaseTVContEncoder):
    def __init__(self, in_channels, h_channels, conv_kernels, paddings, activation,
                 sa_hidden, sa_layer, sa_head, sa_filter_size, sa_kernel_size,
                 sa_dropout, out_channels, max_seq_len, device):
        super(TVConvSAINEncoder, self).__init__(device)
        # convolutions
        self.out_channels = out_channels
        self.activation = get_activation(activation)
        self.conv_initial = nn.Conv1d(in_channels, h_channels, kernel_size=1, padding=0)
        self.conv_layers = nn.ModuleList([
            ConvINBlk(
                h_channels, h_channels, kernel_size=k, padding=p, activation=self.activation)
            for k, p in zip(conv_kernels, paddings)])
        # self-attentions
        n_position = max_seq_len + 1
        d_k = d_v = sa_hidden // sa_head
        self.max_seq_len = max_seq_len
        self.d_model = sa_hidden

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, sa_hidden).unsqueeze(0),
            requires_grad=False)

        self.layer_stack = nn.ModuleList(
            [FFTBlock(
                sa_hidden, sa_head, d_k, d_v, sa_filter_size,
                sa_kernel_size, dropout=sa_dropout)
                for _ in range(sa_layer)])

        self.mu_logs_linear = nn.Conv1d(sa_hidden, 2 * out_channels, kernel_size=1, padding=0)

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
        h = masked_instance_norm(h, lens)
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

        # get distribution parameters
        mu_logs = self.mu_logs_linear(dec_output)
        mu, log_std = torch.split(mu_logs, self.out_channels, dim=1)
        return mu, log_std


class TVConvSABNEncoder(BaseTVContEncoder):
    def __init__(self, in_channels, h_channels, conv_kernels, paddings, activation,
                 sa_hidden, sa_layer, sa_head, sa_filter_size, sa_kernel_size,
                 sa_dropout, out_channels, max_seq_len, device):
        super(TVConvSABNEncoder, self).__init__(device)
        # convolutions
        self.out_channels = out_channels
        self.activation = get_activation(activation)
        self.conv_initial = nn.Conv1d(in_channels, h_channels, kernel_size=1, padding=0)
        self.conv_layers = nn.ModuleList([
            ConvBNBlk(
                h_channels, h_channels, kernel_size=k, padding=p, activation=self.activation)
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
            [FFTBlock(
                sa_hidden, sa_head, d_k, d_v, sa_filter_size,
                sa_kernel_size, dropout=sa_dropout)
                for _ in range(sa_layer)])

        self.mu_logs_linear = nn.Conv1d(sa_hidden, 2 * out_channels, kernel_size=1, padding=0)

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

        # get distribution parameters
        mu_logs = self.mu_logs_linear(dec_output)
        mu, log_std = torch.split(mu_logs, self.out_channels, dim=1)
        return mu, log_std


class TVConvDiscLNEncoder(BaseTVDiscEncoder):
    """
    Encode time-invariant features as discrete sequences
    """

    def __init__(self, in_channels, h_channels, n_class, conv_kernels,
                 paddings, activation, embd_dim, temperature, device):
        super(TVConvDiscLNEncoder, self).__init__(device, n_class, temperature)
        self.activation = get_activation(activation)
        self.conv_initial = nn.Conv1d(in_channels, h_channels, kernel_size=(1,), padding=0)
        self.conv_layers = nn.ModuleList([
            ConvResLNBlk(h_channels, h_channels, kernel_size=k, padding=p, activation=self.activation)
            for k, p in zip(conv_kernels, paddings)])
        self.out_layer_norm = nn.LayerNorm(h_channels)
        self.logit_linear = nn.Conv1d(h_channels, n_class, kernel_size=(1,))
        self.embeds = nn.Parameter(torch.FloatTensor(1, n_class, embd_dim))
        nn.init.uniform_(self.embeds)

    def forward(self, x, lens=None):
        """
        :param x: [batch, channels, time-length]
        :return: mu and logs of shape [batch, out_channels]
        """
        h = self.conv_initial(x)
        for conv in self.conv_layers:
            h = conv(h)
        h = self.activation(h)
        h = self.out_layer_norm(h.permute(0, 2, 1)).permute(0, 2, 1)
        logits = self.logit_linear(h)
        # sample
        samples = self.sample(logits)  # [B, T, C]
        out = torch.matmul(samples, self.embeds).permute(0, 2, 1)  # [B, embd_dim, T]
        return logits, out


class TVConvDiscLNEncoderInd(nn.Module):
    """
    Encode time-invariant features as discrete sequences
    """

    def __init__(self, in_channels, h_channels, n_pitch_class, n_energy_class, conv_kernels,
                 paddings, activation, pit_dim, ene_dim, temperature, device):
        super(TVConvDiscLNEncoderInd, self).__init__()
        self.device = device
        self.n_pitch_class = n_pitch_class
        self.n_energy_class = n_energy_class
        self.temperature = temperature
        self.activation = get_activation(activation)
        self.conv_initial = nn.Conv1d(in_channels, h_channels, kernel_size=(1,), padding=0)
        self.conv_layers = nn.ModuleList([
            ConvResLNBlk(h_channels, h_channels, kernel_size=k, padding=p, activation=self.activation)
            for k, p in zip(conv_kernels, paddings)])
        self.out_layer_norm = nn.LayerNorm(h_channels)
        self.pitch_linear = nn.Conv1d(h_channels, n_pitch_class, kernel_size=(1,))
        self.energy_linear = nn.Conv1d(h_channels, n_energy_class, kernel_size=(1,))
        self.pitch_embeds = nn.Parameter(torch.FloatTensor(1, n_pitch_class, pit_dim))
        self.energy_embeds = nn.Parameter(torch.FloatTensor(1, n_energy_class, ene_dim))
        nn.init.uniform_(self.pitch_embeds)
        nn.init.uniform_(self.energy_embeds)

    def pitch_embed(self, idx):
        """
        :param idx: [B, T]
        :return:
        """
        onehot = F.one_hot(idx, num_classes=self.n_pitch_class)
        return torch.matmul(onehot.float(), self.pitch_embeds).permute(0, 2, 1)

    def energy_embed(self, idx):
        """
        :param idx: [B, T]
        :return:
        """
        onehot = F.one_hot(idx, num_classes=self.n_energy_class)
        return torch.matmul(onehot.float(), self.energy_embeds).permute(0, 2, 1)

    def forward(self, x, lens=None):
        """
        :param x: [batch, channels, time-length]
        :param lens: [batch]
        :return: mu and logs of shape [batch, out_channels]
        """
        h = self.conv_initial(x)
        for conv in self.conv_layers:
            h = conv(h)
        h = self.activation(h)
        h = self.out_layer_norm(h.permute(0, 2, 1)).permute(0, 2, 1)
        pitch_logits = self.pitch_linear(h)
        energy_logits = self.energy_linear(h)
        # sample
        pitch_samples = self.sample(pitch_logits)  # [B, T, C]
        energy_samples = self.sample(energy_logits)  # [B, T, C]
        pitch_embds = torch.matmul(pitch_samples, self.pitch_embeds).permute(0, 2, 1)  # [B, embd_dim, T]
        energy_embds = torch.matmul(energy_samples, self.energy_embeds).permute(0, 2, 1)  # [B, embd_dim, T]
        return pitch_logits, energy_logits, pitch_embds, energy_embds

    def sample(self, logits):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.
        Parameters
        ----------
        logits : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D, T)
        """
        logits = logits.permute(0, 2, 1)  # [B, T, D]
        samples = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        return samples

    @staticmethod
    def kl_divergence_with_prior(logits, prior, lens, EPS=1e-12):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.
        Parameters
        ----------
        logits : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D, T)
        prior : torch.Tensor
            shape (N, T)
        """
        prior = F.one_hot(prior, num_classes=logits.size(1))  # [B, T, D]
        alpha = F.softmax(logits, dim=1)
        B, D, T = alpha.size()
        if lens is not None:
            mask = sequence_mask(lens).to(alpha.dtype)
        else:
            mask = torch.ones([B, T], dtype=alpha.dtype, device=alpha.device)
        alpha = alpha.permute(0, 2, 1).reshape([B * T, D])
        prior = prior.reshape([B * T, D])
        kl_loss = torch.sum(alpha * (torch.log(alpha + EPS) - torch.log(prior + EPS)), dim=1)  # [B * T]
        kl_loss = kl_loss.reshape([B, T])
        kl_loss = torch.mean(torch.sum(kl_loss * mask, dim=1) / torch.sum(mask, dim=1))
        return kl_loss

    def kl_divergence_with_uniform(self, logits, lens, EPS=1e-12):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.
        Parameters
        ----------
        logits : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D, T)
        """
        alpha = F.softmax(logits, dim=1)
        B, D, T = alpha.size()
        if lens is not None:
            mask = sequence_mask(lens).to(alpha.dtype)
        else:
            mask = torch.ones([B, T], dtype=alpha.dtype, device=alpha.device)
        alpha = alpha.permute(0, 2, 1).reshape([B * T, D])
        log_dim = torch.log(torch.ones([1], dtype=alpha.dtype, device=self.device) * D)
        # Calculate negative entropy of each row
        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
        # Take mean of negative entropy across batch
        neg_entropy = neg_entropy.reshape([B, T])
        mean_neg_entropy = torch.mean(
            torch.sum(neg_entropy * mask, dim=1) / torch.sum(mask, dim=1))
        # KL loss of alpha with uniform categorical variable
        kl_loss = log_dim + mean_neg_entropy
        return kl_loss


class TVConvDiscLNEncoderJoint(nn.Module):
    """
    Encode time-invariant features as discrete sequences
    """
    def __init__(self, in_channels, h_channels, n_pitch_class, n_energy_class,
                 conv_kernels, paddings, activation, pit_dim, ene_dim, temperature, device):
        super(TVConvDiscLNEncoderJoint, self).__init__()
        self.device = device
        self.n_pitch_class = n_pitch_class
        self.n_energy_class = n_energy_class
        self.temperature = temperature
        self.activation = get_activation(activation)
        self.conv_initial = nn.Conv1d(in_channels, h_channels, kernel_size=(1,), padding=0)
        self.conv_layers = nn.ModuleList([
            ConvResLNBlk(h_channels, h_channels, kernel_size=k, padding=p, activation=self.activation)
            for k, p in zip(conv_kernels, paddings)])
        self.out_layer_norm = nn.LayerNorm(h_channels)
        self.logit_linear = nn.Conv1d(h_channels, n_pitch_class * n_energy_class, kernel_size=(1,))
        self.pitch_embeds = nn.Parameter(torch.FloatTensor(1, n_pitch_class, pit_dim))
        self.energy_embeds = nn.Parameter(torch.FloatTensor(1, n_energy_class, ene_dim))
        nn.init.uniform_(self.pitch_embeds)
        nn.init.uniform_(self.energy_embeds)

    def pitch_embed(self, idx):
        """
        :param idx: [B, T]
        :return:
        """
        onehot = F.one_hot(idx, num_classes=self.n_pitch_class)
        return torch.matmul(onehot.float(), self.pitch_embeds).permute(0, 2, 1)

    def energy_embed(self, idx):
        """
        :param idx: [B, T]
        :return:
        """
        onehot = F.one_hot(idx, num_classes=self.n_energy_class)
        return torch.matmul(onehot.float(), self.energy_embeds).permute(0, 2, 1)

    def forward(self, x, lens=None):
        """
        :param x: [batch, channels, time-length]
        :param lens: [batch]
        :return: mu and logs of shape [batch, out_channels]
        """
        h = self.conv_initial(x)
        for conv in self.conv_layers:
            h = conv(h)
        h = self.activation(h)
        h = self.out_layer_norm(h.permute(0, 2, 1)).permute(0, 2, 1)
        logits = self.logit_linear(h)
        pitch_samples, energy_samples = self.sample(logits)  # [B, T, C]
        pitch_embds = torch.matmul(pitch_samples, self.pitch_embeds).permute(0, 2, 1)  # [B, embd_dim, T]
        energy_embds = torch.matmul(energy_samples, self.energy_embeds).permute(0, 2, 1)  # [B, embd_dim, T]
        return logits, pitch_embds, energy_embds

    def sample(self, logits):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.
        Parameters
        ----------
        logits : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D, T)
        """
        logits = logits.permute(0, 2, 1)  # [B, T, D]
        samples = F.gumbel_softmax(
            logits, tau=self.temperature, hard=True).reshape(
            [logits.size(0), logits.size(1), self.n_pitch_class, self.n_energy_class])
        # [B, T, n_pitch_class, n_energy_class]
        pitch_samples = torch.sum(samples, dim=3)  # [B, T, n_pitch_class]
        energy_samples = torch.sum(samples, dim=2)  # [B, T, n_energy_class]
        return pitch_samples, energy_samples

    def kl_divergence_between(self, post_dist, prior_dist, lens, EPS=1e-12):
        """
        :param post_dist: [B, T, D]
        :param prior_dist: [B, T, D]
        :param lens: [B]
        :return:
        """
        B, T, D = post_dist.size()
        if lens is not None:
            mask = sequence_mask(lens).to(post_dist.dtype)
        else:
            mask = torch.ones([B, T], dtype=post_dist.dtype, device=post_dist.device)
        post = post_dist.reshape([B * T, D])
        prior = prior_dist.reshape([B * T, D])
        kl_loss = torch.sum(post * (torch.log(post + EPS) - torch.log(prior + EPS)), dim=1)  # [B * T]
        kl_loss = kl_loss.reshape([B, T])
        kl_loss = torch.mean(torch.sum(kl_loss * mask, dim=1) / torch.sum(mask, dim=1))
        return kl_loss

    def kl_divergence_with_prior(self, logits, pit_label, ene_label, lens, EPS=1e-12):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.
        Parameters
        ----------
        logits : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D, T)
        pit_label : torch.Tensor
            shape (N, T)
        ene_label : torch.Tensor
        shape (N, T)
        """
        # joint kl
        jnt_label = pit_label * self.n_energy_class + ene_label
        jnt_prior = F.one_hot(jnt_label, num_classes=self.n_pitch_class * self.n_energy_class)  # [B, T, D']
        jnt_post = F.softmax(logits, dim=1)  # [B, D', T]
        jnt_post = jnt_post.permute(0, 2, 1)  # [B, T, D']
        jnt_kl = self.kl_divergence_between(jnt_post, jnt_prior, lens, EPS)

        jnt_post = jnt_post.reshape(
            [jnt_post.size(0), jnt_post.size(1), self.n_pitch_class, self.n_energy_class])
        jnt_prior = jnt_prior.reshape(
            [jnt_prior.size(0), jnt_prior.size(1), self.n_pitch_class, self.n_energy_class])

        # pitch kl
        pit_post = torch.sum(jnt_post, dim=3)  # marginalize over energy axis
        pit_prior = torch.sum(jnt_prior, dim=3)  # [B, T, D_p]
        pit_kl = self.kl_divergence_between(pit_post, pit_prior, lens, EPS)

        # energy kl
        ene_post = torch.sum(jnt_post, dim=2)  # marginalize over pitch axis
        ene_prior = torch.sum(jnt_prior, dim=2)  # [B, T, D_e]
        ene_kl = self.kl_divergence_between(ene_post, ene_prior, lens, EPS)
        return jnt_kl, pit_kl, ene_kl

    def kl_divergence_with_uniform(self, logits, lens, EPS=1e-12):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.
        Parameters
        ----------
        logits : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D, T)
        """
        alpha = F.softmax(logits, dim=1)
        B, D, T = alpha.size()
        if lens is not None:
            mask = sequence_mask(lens).to(alpha.dtype)
        else:
            mask = torch.ones([B, T], dtype=alpha.dtype, device=alpha.device)
        alpha = alpha.permute(0, 2, 1).reshape([B * T, D])
        log_dim = torch.log(torch.ones([1], dtype=alpha.dtype, device=self.device) * D)
        # Calculate negative entropy of each row
        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
        # Take mean of negative entropy across batch
        neg_entropy = neg_entropy.reshape([B, T])
        mean_neg_entropy = torch.mean(
            torch.sum(neg_entropy * mask, dim=1) / torch.sum(mask, dim=1))
        # KL loss of alpha with uniform categorical variable
        kl_loss = log_dim + mean_neg_entropy
        return kl_loss


class TVConvDiscBNEncoder(BaseTVDiscEncoder):
    """
    Encode time-invariant features as discrete sequences
    """

    def __init__(self, in_channels, h_channels, n_class, conv_kernels,
                 paddings, activation, embd_dim, temperature, device):
        super(TVConvDiscBNEncoder, self).__init__(device, n_class, temperature)
        self.activation = get_activation(activation)
        self.conv_initial = nn.Conv1d(in_channels, h_channels, kernel_size=(1,), padding=0)
        self.conv_layers = nn.ModuleList([
            ConvResBNBlk(h_channels, h_channels, kernel_size=k, padding=p, activation=self.activation)
            for k, p in zip(conv_kernels, paddings)])
        self.out_batch_norm = MaskedBatchNorm1d(h_channels)
        self.logit_linear = nn.Conv1d(h_channels, n_class, kernel_size=(1,))
        self.embeds = nn.Parameter(torch.FloatTensor(1, n_class, embd_dim))
        nn.init.uniform_(self.embeds)

    def forward(self, x, lens=None):
        """
        :param x: [batch, channels, time-length]
        :param lens: [batch]
        :return: mu and logs of shape [batch, out_channels]
        """
        h = self.conv_initial(x)
        for conv in self.conv_layers:
            h = conv(h, lens)
        h = self.activation(h)
        h = self.out_batch_norm(h, lens)
        logits = self.logit_linear(h)
        # sample
        samples = self.sample(logits)  # [B, T, C]
        out = torch.matmul(samples, self.embeds).permute(0, 2, 1)  # [B, embd_dim, T]
        return logits, out


class TVConvSADiscINEncoder(BaseTVDiscEncoder):
    def __init__(self, in_channels, h_channels, conv_kernels, paddings, activation,
                 sa_hidden, sa_layer, sa_head, sa_filter_size, sa_kernel_size,
                 sa_dropout, max_seq_len, n_class, embd_dim, temperature, device):
        super(TVConvSADiscINEncoder, self).__init__(device, n_class, temperature)
        # convolutions
        self.activation = get_activation(activation)
        self.conv_initial = nn.Conv1d(in_channels, h_channels, kernel_size=1, padding=0)
        self.conv_layers = nn.ModuleList([
            ConvINBlk(
                h_channels, h_channels, kernel_size=k, padding=p, activation=self.activation)
            for k, p in zip(conv_kernels, paddings)])
        # self-attentions
        n_position = max_seq_len + 1
        d_k = d_v = sa_hidden // sa_head
        self.max_seq_len = max_seq_len
        self.d_model = sa_hidden

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, sa_hidden).unsqueeze(0),
            requires_grad=False)

        self.layer_stack = nn.ModuleList(
            [FFTBlock(
                sa_hidden, sa_head, d_k, d_v, sa_filter_size,
                sa_kernel_size, dropout=sa_dropout)
                for _ in range(sa_layer)])
        self.disc_dim = n_class
        self.temperature = temperature
        self.logit_linear = nn.Conv1d(sa_hidden, n_class, kernel_size=(1,))
        self.embeds = nn.Parameter(torch.FloatTensor(1, n_class, embd_dim))
        nn.init.uniform_(self.embeds)

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
        h = masked_instance_norm(h, lens)
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
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
        dec_output = dec_output.permute(0, 2, 1).contiguous()

        # get distribution parameters
        logits = self.logit_linear(dec_output)

        # sample
        samples = self.sample(logits)  # [B, T, C]
        out = torch.matmul(samples, self.embeds).permute(0, 2, 1)  # [B, embd_dim, T]
        return logits, out


class TVConvSADiscBNEncoder(BaseTVDiscEncoder):
    def __init__(self, in_channels, h_channels, conv_kernels, paddings, activation,
                 sa_hidden, sa_layer, sa_head, sa_filter_size, sa_kernel_size,
                 sa_dropout, max_seq_len, n_class, embd_dim, temperature, device):
        super(TVConvSADiscBNEncoder, self).__init__(device, n_class, temperature)
        # convolutions
        self.activation = get_activation(activation)
        self.conv_initial = nn.Conv1d(in_channels, h_channels, kernel_size=1, padding=0)
        self.conv_layers = nn.ModuleList([
            ConvBNBlk(
                h_channels, h_channels, kernel_size=k, padding=p, activation=self.activation)
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
            [FFTBlock(
                sa_hidden, sa_head, d_k, d_v, sa_filter_size,
                sa_kernel_size, dropout=sa_dropout)
                for _ in range(sa_layer)])
        self.disc_dim = n_class
        self.temperature = temperature
        self.logit_linear = nn.Conv1d(sa_hidden, n_class, kernel_size=(1,))
        self.embeds = nn.Parameter(torch.FloatTensor(1, n_class, embd_dim))
        nn.init.uniform_(self.embeds)

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
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
        dec_output = dec_output.permute(0, 2, 1).contiguous()

        # get distribution parameters
        logits = self.logit_linear(dec_output)

        # sample
        samples = self.sample(logits)  # [B, T, C]
        out = torch.matmul(samples, self.embeds).permute(0, 2, 1)  # [B, embd_dim, T]
        return logits, out
