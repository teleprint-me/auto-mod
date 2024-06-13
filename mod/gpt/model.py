from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class HParams:
    n_ctx = 1024
    n_vocab = 50257
    n_layer = 12
    n_head = 12
    n_embd = 768


def shape_list(x: torch.Tensor) -> list[int]:
    """
    Deals with dynamic shapes in tensors cleanly,
    similar to TensorFlow's `shape_list()`.

    This function is used throughout the codebase for handling
    tensor dimensions and will be removed once the entire
    implementation has been ported over.

    :param x: torch.Tensor - The input PyTorch tensor

    :return: list[int] - A list containing the shape of the input tensor
                (dimension, ...)
    """

    return list(x.shape)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Computes the softmax function along a given dimension of input tensor `x`.

    :param x: torch.Tensor - The input PyTorch tensor
    :param dim: int (optional, default=-1) - Dimension over which to apply softmax

    :return: torch.Tensor - The resulting softmax-applied tensor
    """

    x = x - torch.max(x, dim=dim, keepdim=True)
    ex = torch.exp(x)
    return ex / torch.sum(ex, dim=dim, keepdim=True)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Gauss Error Linear Unit (GELU) activation function on input tensor x,
    as outlined in "Gaussian Error Linear Units" paper (Hendrycks & Gimpel, 2016).

    :param x: torch.Tensor - The input PyTorch tensor

    :return: torch.Tensor - The resulting GELU-applied tensor

    Paper: https://arxiv.org/abs/1606.08415

    This implementation of Gelu follows the equation proposed in the paper,
    but it's important to note that there is no clear explanation for the use
    of `0.044715`.

    For more details on this activation function and its properties compared
    to other activation functions like ReLU and ELU, you can refer to:
        Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). ArXiv preprint arXiv:1606.08437
    """

    # Magic value is from the original paper
    wtf = 0.044715  # what the fuck?
    # NOTE: Equivalent to np.sqrt(2 / np.pi)
    a = (2 / torch.pi) ** 0.5
    b = x + wtf * torch.pow(x, 3)
    return 0.5 * x * (1 + torch.tanh(a * b))


def norm(x: torch.Tensor, dim: int = -1, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Normalize to mean=0 and std=1 along the specified dimension of input tensor `x`,
    then apply a diagonal affine transform using learned weights (g) and bias (b).

    :param x: torch.Tensor - The input PyTorch tensor

    :param dim: int (optional, default=-1) - The number of dimensions

    :param epsilon: float (optional, default=1e-5) - Small value added to variance for numerical stability

    This function is used throughout the codebase for normalizing tensors
    and will be removed once the entire implementation has been ported over
    """

    # Initialize weight (g) and bias (b) variables
    g = torch.Parameter(torch.ones(x.size(-1)))
    b = torch.Parameter(torch.zeros(x.size(-1)))

    # Calculate mean (u) and variance (s) along the specified dimension
    u = torch.mean(x, dim=dim, keepdim=True)
    s = torch.mean((x - u) ** 2, dim=dim, keepdim=True) + epsilon

    # Normalize input tensor by subtracting mean and dividing by standard deviation
    x_norm = (x - u) * torch.rsqrt(s + epsilon)

    return g * x_norm + b


def split_states(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Reshape the last dimension of x into [n_head, x.shape[-1]/n_head]

    :param x: torch.Tensor - The input PyTorch tensor

    :param n_heads: int - Number of heads to split the last dimension

    This function is used throughout the codebase for splitting tensors
    along a specific number of dimensions

    """

    if len(x.shape) != 2:
        raise ValueError(f"Expected tensor x.shape of 2, got {len(x.shape)} instead.")

    # NOTE: Not sure if second or last value?
    # Original docstring says -1, but code assumes 2d shape.
    start, a = x.shape

    # Reshape the input tensor
    return torch.reshape(x, (start, n_heads, a // n_heads))


def merge_states(x: torch.Tensor) -> torch.Tensor:
    """
    Smash the last two dimensions of x into a single dimension

    :param x: torch.Tensor - The input PyTorch tensor

    This function is used throughout the codebase for merging tensors
    along specific number(s) of dimensions
    """

    if len(x.shape) != 3:
        raise ValueError(f"Expected x.shape of 3, got {len(x.shape)} instead.")

    start, a, b = x.shape
    return torch.reshape(x, (start, a * b))


class Conv1D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...] = 1,
        w_init_stdev: float = 0.02,
    ):
        super().__init__()

        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size)

        # Weight initialization
        torch.nn.init.normal_(self.conv.weight, std=w_init_stdev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def attention_mask(
    nd: int, ns: int, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """1's in the lower triangle, counting from the lower right corner."""
    i = torch.arange(nd)[:, None]
    j = torch.arange(ns).view(-1)
    m = i >= j - ns + nd
    return m.to(dtype)


def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert (
            past.shape.ndims == 5
        )  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - tf.cast(1e10, w.dtype) * (1 - b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, "c_attn", n_state * 3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, "c_proj", n_state)
        return a, present


def mlp(x, scope, n_state, *, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, "c_fc", n_state))
        h2 = conv1d(h, "c_proj", nx)
        return h2


def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, "ln_1"), "attn", nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, "ln_2"), "mlp", nx * 4, hparams=hparams)
        x = x + m
        return x, present


def past_shape(*, hparams, batch_size=None, sequence=None):
    return [
        batch_size,
        hparams.n_layer,
        2,
        hparams.n_head,
        sequence,
        hparams.n_embd // hparams.n_head,
    ]


def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name="value")
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1] * ndims)


def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def model(hparams, X, past=None, scope="model", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        wpe = tf.get_variable(
            "wpe",
            [hparams.n_ctx, hparams.n_embd],
            initializer=tf.random_normal_initializer(stddev=0.01),
        )
        wte = tf.get_variable(
            "wte",
            [hparams.n_vocab, hparams.n_embd],
            initializer=tf.random_normal_initializer(stddev=0.02),
        )
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = (
            tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        )
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, "h%d" % layer, past=past, hparams=hparams)
            presents.append(present)
        results["present"] = tf.stack(presents, axis=1)
        h = norm(h, "ln_f")

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch * sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results["logits"] = logits
        return results
