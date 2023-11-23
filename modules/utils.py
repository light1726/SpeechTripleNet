import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSampleEncBlk(nn.Module):
    """
    1D convolutional block for time-invariant feature extraction
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 activation, pooling_kernel, pooling_type='mean'):
        # pooling_type in ['mean', 'max']
        super(DownSampleEncBlk, self).__init__()
        self.layer_norm0 = nn.LayerNorm(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.layer_norm1 = nn.LayerNorm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv_sc = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        if pooling_type == 'mean':
            self.downsample = nn.AvgPool1d(kernel_size=pooling_kernel)
        elif pooling_type == 'max':
            self.downsample = nn.MaxPool1d(kernel_size=pooling_kernel)
        else:
            raise ValueError('Unrecognized pooling type: {}'.format(pooling_type))
        self.activation = activation

    def forward(self, x):
        """
        :param x: [batch, channels, time-length]
        :return:
        """
        h = self.activation(x)
        h = self.layer_norm0(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.activation(self.conv1(h))
        h = self.layer_norm1(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.conv2(h)
        h = self.downsample(h)
        sc = self.downsample(self.conv_sc(x))
        out = h + sc
        return out


class ConvINBlk(nn.Module):
    """
    1D convolutional block for time-variant feature extraction with instance normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, activation):
        super(ConvINBlk, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.activation = activation

    def forward(self, x, lens):
        """
        :param x: [batch, channels, time-length]
        :param lens: [batch, time-length]
        :return:
        """
        h = self.activation(x)
        h = masked_instance_norm(h, lens)
        h = self.activation(self.conv1(h))
        h = masked_instance_norm(h, lens)
        h = self.conv2(h)
        return h


class ConvBNBlk(nn.Module):
    """
    1D convolutional block for time-variant feature extraction with batch normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, activation):
        super(ConvBNBlk, self).__init__()
        self.batch_norm0 = MaskedBatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.batch_norm1 = MaskedBatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.activation = activation

    def forward(self, x, lens):
        """
        :param x: [batch, channels, time-length]
        :param lens: [batch]
        :return:
        """
        h = self.activation(x)
        h = self.batch_norm0(h, lens)
        h = self.activation(self.conv1(h))
        h = self.batch_norm1(h, lens)
        h = self.conv2(h)
        return h


class ConvLNBlk(nn.Module):
    """
    1D convolutional block for time-variant feature extraction
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, activation):
        super(ConvLNBlk, self).__init__()
        self.layer_norm0 = nn.LayerNorm(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.layer_norm1 = nn.LayerNorm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.activation = activation

    def forward(self, x):
        """
        :param x: [batch, channels, time-length]
        :return:
        """
        h = self.activation(x)
        h = self.layer_norm0(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.activation(self.conv1(h))
        h = self.layer_norm1(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.conv2(h)
        return h


class ConvResLNBlk(nn.Module):
    """
    1D convolutional block for decoder
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, activation):
        super(ConvResLNBlk, self).__init__()
        self.layer_norm0 = nn.LayerNorm(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.layer_norm1 = nn.LayerNorm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv_sc = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.activation = activation

    def forward(self, x):
        """
        :param x: [batch, channels, time-length]
        :return:
        """
        h = self.activation(x)
        h = self.layer_norm0(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.activation(self.conv1(h))
        h = self.layer_norm1(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.conv2(h)
        sc = self.conv_sc(x)
        out = h + sc
        return out


class ConvResINBlk(nn.Module):
    """
    1D convolutional block for decoder
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, activation):
        super(ConvResINBlk, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv_sc = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.activation = activation

    def forward(self, x, lens):
        """
        :param x: [batch, channels, time-length]
        :return:
        """
        h = self.activation(x)
        h = masked_instance_norm(h, lens)
        h = self.activation(self.conv1(h))
        h = masked_instance_norm(h, lens)
        h = self.conv2(h)
        sc = self.conv_sc(x)
        out = h + sc
        return out


class ConvResBNBlk(nn.Module):
    """
    1D convolutional block for decoder
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, activation):
        super(ConvResBNBlk, self).__init__()
        self.batch_norm0 = MaskedBatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.batch_norm1 = MaskedBatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv_sc = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.activation = activation

    def forward(self, x, lens):
        """
        :param x: [batch, channels, time-length]
        :return:
        """
        h = self.activation(x)
        h = self.batch_norm0(h, lens)
        h = self.activation(self.conv1(h))
        h = self.batch_norm1(h, lens)
        h = self.conv2(h)
        sc = self.conv_sc(x)
        out = h + sc
        return out


class VectorQuantizerEMA(nn.Module):
    """
    Adapted from https://github.com/swasun/VQ-VAE-Images/blob/master/src/vector_quantizer_ema.py
    Args:
        embedding_dim: integer representing the dimensionality of the tensors in the
            quantized space. Inputs to the modules must be in this format as well.
        num_embeddings: integer, the number of vectors in the quantized space.
            commitment_cost: scalar which controls the weighting of the loss terms (see
            equation 4 in the paper).
        decay: float, decay for the moving averages.
        epsilon: small float constant to avoid numerical instability.
    """

    def __init__(self, device, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._device = device

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        """
        Connects the module to some inputs.
        Args:
            inputs: Tensor, final dimension must be equal to embedding_dim. All other
                leading dimensions will be flattened and treated as a large batch.

        Returns:
            dict containing the following keys and values:
            quantize: Tensor containing the quantized version of the input.
            loss: Tensor containing the loss to optimize.
            perplexity: Tensor containing the perplexity of the encodings.
            encodings: Tensor containing the discrete encodings, ie which element
                of the quantized space each input element was mapped to.
            encoding_indices: Tensor containing the discrete encoding indices, ie
                which element of the quantized space each input element was mapped to.
        """
        # Calculate distances
        distances = (torch.sum(inputs ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(inputs, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(self._device)
        encodings.scatter_(1, encoding_indices, 1)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n
            )

            dw = torch.matmul(encodings.t(), inputs)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight)

        # Loss
        e_latent_loss = torch.mean(0.5 * torch.sum((quantized.detach() - inputs) ** 2, dim=1))
        loss = self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings

    @property
    def embedding(self):
        return self._embedding


def masked_instance_norm(x, lens, eps=1e-5):
    """
    x : [B, C, T]
    mask: [B, T]
    x of shape: [batch_size (N), num_objects (L), features(C)]
    mask of shape: [batch_size (N), num_objects (L)]
    """
    if lens is not None:
        mask = sequence_mask(lens, max_length=x.size(2))
    else:
        B, T = x.size(0), x.size(2)
        mask = torch.ones([B, T], dtype=torch.bool, device=x.device)
    mask = mask.float().unsqueeze(1)  # [B, 1, T]
    mean = (torch.sum(x * mask, 2) / torch.sum(mask, 2))  # [B, C]
    mean = mean.detach()
    var_term = ((x - mean.unsqueeze(2).expand_as(x)) * mask) ** 2  # [B, C, T]
    var = (torch.sum(var_term, 2) / torch.sum(mask, 2))  # [B, C]
    var = var.detach()
    mean_reshaped = mean.unsqueeze(2)  # [B, C, T]
    var_reshaped = var.unsqueeze(2)  # [B, C, T]
    ins_norm = (x - mean_reshaped) / torch.sqrt(var_reshaped + eps)  # (N, L, C)
    return ins_norm


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def get_activation(act_str):
    return {'relu': F.relu, 'leaky_relu': F.leaky_relu}[act_str]


"""
below masked version of 1D batch normalization is borrowed from
https://gist.github.com/amiasato/902fc14afa37a7537386f7b0c5537741
"""


def lengths_to_mask(lengths, max_len=None, dtype=None):
    """
    Converts a "lengths" tensor to its binary mask representation.

    Based on: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397

    :lengths: N-dimensional tensor
    :returns: N*max_len dimensional tensor. If max_len==None, max_len=max(lengtsh)
    """
    assert len(lengths.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or lengths.max().item()
    mask = torch.arange(
        max_len,
        device=lengths.device,
        dtype=lengths.dtype) \
               .expand(len(lengths), max_len) < lengths.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=lengths.device)
    return mask


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """
    Masked verstion of the 1D Batch normalization.

    Based on: https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py

    Receives a N-dim tensor of sequence lengths per batch element
    along with the regular input for masking.

    Check pytorch's BatchNorm1d implementation for argument details.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MaskedBatchNorm1d, self).__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats
        )

    def forward(self, inp, lengths):
        self._check_input_dim(inp)

        exponential_average_factor = 0.0

        # We transform the mask into a sort of P(inp) with equal probabilities
        # for all unmasked elements of the tensor, and 0 probability for masked
        # ones.
        mask = lengths_to_mask(lengths, max_len=inp.shape[-1], dtype=inp.dtype)
        n = mask.sum()
        mask = mask / n
        mask = mask.unsqueeze(1).expand(inp.shape)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training and n > 1:
            # Here lies the trick. Using Var(X) = E[X^2] - E[X]^2 as the biased
            # variance, we do not need to make any tensor shape manipulation.
            # mean = E[X] is simply the sum-product of our "probability" mask with the input...
            mean = (mask * inp).sum([0, 2])
            # ...whereas Var(X) is directly derived from the above formulae
            # This should be numerically equivalent to the biased sample variance
            var = (mask * inp ** 2).sum([0, 2]) - mean ** 2
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # Update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        inp = (inp - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
        if self.affine:
            inp = inp * self.weight[None, :, None] + self.bias[None, :, None]

        return inp
