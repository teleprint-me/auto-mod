from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass
class Config:
    n_blocks: int = 256
    n_vocab: int = 30000
    n_layer: int = 8
    n_head: int = 4
    n_embed: int = 264


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True)
    ex = torch.exp(x)
    return ex / torch.sum(ex, dim=dim, keepdim=True)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return (
        0.5
        * x
        * (1 + torch.tanh(((2 / np.pi) ** 0.5) * (x + 0.044715 * torch.pow(x, 3))))
    )


class Norm(torch.nn.Module):
    def __init__(self, n_state: Sequence[int], dim: int = -1, eps: float = 1e-5):
        super().__init__()

        self.g = torch.nn.Parameter(torch.ones((n_state)))
        self.b = torch.nn.Parameter(torch.zeros((n_state)))

        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.mean(x, dim=self.dim, keepdim=True)
        s = torch.sqrt(torch.mean(torch.square(x - u), dim=self.dim, keepdim=True))

        return self.g * (x - u) / (s + self.eps) + self.b


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


# TODO: This function will be removed once the code is fully ported over.
# def shape_list(x):
#     """Deal with dynamic shape in tensorflow cleanly."""
#     static = x.shape.as_list()
#     dynamic = tf.shape(x)
#     return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def split_states(x: torch.Tensor, n) -> torch.Tensor:
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = x.shape
    return x.view(*start, n, m // n)


def merge_states(x: torch.Tensor) -> torch.Tensor:
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = x.shape
    return x.view(*start, a * b)


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
