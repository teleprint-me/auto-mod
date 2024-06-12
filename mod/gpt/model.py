import numpy as np
import torch
from tensorflow.contrib.training import HParams
from dataclasses import dataclass


@dataclass
class Config:
    n_blocks: int = 256
    n_vocab: int = 30000
    n_layer: int = 8
    n_head: int = 4
    n_embed: int = 264


# TODO: This function should most likely be removed.
# NOTE:
# In TensorFlow, `x.shape.as_list()` returns a list of integers or None entries (for unknown dimensions) representing the shape of tensor `x`.
# However, because TensorFlow operates in a static computational graph and some shapes may be only known at runtime, this method can return partially defined shapes that contain both concrete values and symbols for unknown ones.
# The `shape_list` function is used to handle partially-defined shapes cleanly by returning the dynamic shape (obtained using `tf.shape(x)`) whenever a dimension in `static` has an unspecified value of None, otherwise it returns the statically known size from `as_list()`.
# In PyTorch, we don't have this issue as all shapes are known at runtime and can be accessed directly using Python indexing (e.g., `x.shape[i]`).
# So we might not need an equivalent function in the ported codebase for handling dynamically-shaped tensors with TensorFlow, unless there's some specific part of the original code where it's necessary to deal with dynamic shapes explicitly (which seems unlikely given what we know about GPT-2 models so far).
# def shape_list(x):
#     """Deal with dynamic shape in tensorflow cleanly."""
#     static = x.shape.as_list()
#     dynamic = tf.shape(x)
#     return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def softmax(x: torch.Tensor, dim: int = -1):
    x = x - torch.max(x, dim=dim, keepdim=True)
    ex = torch.exp(x)
    return ex / torch.sum(ex, dim=dim, keepdim=True)


def gelu(x: torch.Tensor):
    return (
        0.5
        * x
        * (1 + torch.tanh(((2 / np.pi) ** 0.5) * (x + 0.044715 * torch.pow(x, 3))))
    )


class Norm(torch.nn.Module):
    def __init__(self, n_state, dim: int = -1, eps=1e-5):
        super().__init__()

        self.g = torch.nn.Parameter(torch.ones((n_state)))
        self.b = torch.nn.Parameter(torch.zeros((n_state)))

        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor):
        u = torch.mean(x, dim=-1, keepdim=True)
        s = torch.sqrt(torch.mean(torch.square(x - u), dim=self.dim, keepdim=True))

        return self.g * (x - u) / (s + self.eps) + self.b


def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m // n])


def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a * b])


def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable(
            "w",
            [1, nx, nf],
            initializer=tf.random_normal_initializer(stddev=w_init_stdev),
        )
        b = tf.get_variable("b", [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(
            tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])) + b,
            start + [nf],
        )
        return c


def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


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
