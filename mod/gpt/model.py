import torch
from config import Config


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
    batches, sequence = x.shape

    # Reshape the input tensor
    return torch.reshape(x, (batches, n_heads, sequence // n_heads))


def merge_states(x: torch.Tensor) -> torch.Tensor:
    """
    Smash the last two dimensions of x into a single dimension

    :param x: torch.Tensor - The input PyTorch tensor

    This function is used throughout the codebase for merging tensors
    along specific number(s) of dimensions
    """

    if len(x.shape) != 3:
        raise ValueError(f"Expected x.shape of 3, got {len(x.shape)} instead.")

    batches, sequence, features = x.shape
    return torch.reshape(x, (batches, sequence * features))


def conv1d(x: torch.Tensor, nf: int, w_init_stdev: float = 0.02) -> torch.Tensor:
    """
    Apply 1D convolution to the input tensor `x`, with specified number of filters (nf),
    and initializer standard deviation (w_init_stdev).

    :param x: torch.Tensor - The input PyTorch tensor
    :param nf: int - Number of filters
    :param w_init_stdev: float (optional, default=0.02) - Initialize standard deviation

    This function is used throughout the codebase for applying 1D convolutions
    """

    if len(x.shape) != 2:
        raise ValueError(f"Expected x.shape of 2, got {len(x.shape)} instead.")

    _, nx = list(x.shape)
    # Initialize weight and bias variables
    w = torch.empty((1, nx, nf))
    w.normal_(mean=0.0, std=w_init_stdev)

    b = torch.zeros((nf,))

    # Perform convolution and bias addition
    c = torch.conv1d(x, weight=w, bias=b, padding=0)

    return c


def attention_mask(
    nd: int, ns: int, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """
    1's in the lower triangle, counting from the lower right corner.

    This function generates an attention mask used in self-attention mechanisms
    to ensure that the model does not attend to future positions when processing
    sequences of varying lengths. The mask is a 2D tensor with dimensions (nd, ns),
    where nd represents the number of elements in one sequence and ns corresponds
    to the total number of elements across all sequences combined.

    :param nd: int - Number of elements in each input sequence
    :param ns: int - Total number of elements across all input sequences
    :param dtype: torch.dtype (optional, default=torch.bfloat16) - The data type for the attention mask tensor

    :return: torch.Tensor - A 2D tensor with dimensions (nd, ns), where values are set to 1 along the lower triangle counting from the bottom right corner
    """

    i = torch.arange(nd)[:, None]
    j = torch.arange(ns).view(-1)
    m = i >= j - ns + nd
    return m.to(dtype)


# multi-head attention
def attn(
    x: torch.Tensor,
    n_state: int,
    past: None | torch.Tensor,
    hparams: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.shape == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None and len(past.shape) != 5:
        raise ValueError(
            "If provided, past should have a shape of [batch_size, 2, num_heads, sequence_length, embed_dim]."
        )

    def split_heads(tensor: torch.Tensor) -> torch.Tensor:
        """Reshape the last dimension of x into [n_head, x.shape[-1]/n_head]"""

        return split_states(tensor).T.permute([0, 2, 1, 3])

    def merge_heads(tensor: torch.Tensor) -> torch.Tensor:
        """Smash the last two dimensions of x into a single dimension"""

        return merge_states(tensor.T.permute([0, 2, 1, 3]))

    def mask_attn_weights(w: torch.Tensor) -> torch.Tensor:
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = list(w.shape)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = torch.reshape(b, [1, 1, nd, ns])
        w = w * b - w.to(w.dtype) * (1 - b)
        return w

    def multihead_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        # q, k, v have shape [batch, heads, sequence, features]
        w = torch.matmul(q, k.T)
        w = w * torch.rsqrt(v.to(w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w, w.dim())
        a = torch.matmul(w, v)
        return a

    # NOTE: torch.squeeze and torch.unqueeze are not directly substitutable
    # for tf.stack and tf.unstack. They should be okay for handling singleton
    # dimensions, but will prove to be problematic when batches are included.

    attend = conv1d(x, nf=n_state * 3)
    q, k, v = map(split_heads, torch.split(attend, 3, dim=2))
    present = torch.squeeze([k, v], dim=1)
    if past is not None:
        pk, pv = torch.unsqueeze(past, dim=1)
        k = torch.concat([pk, k], dim=-2)
        v = torch.concat([pv, v], dim=-2)
    a = multihead_attn(q, k, v)
    a = merge_heads(a)
    project = conv1d(a, nf=n_state)
    return project, present


# multi-layer perceptron
def mlp(x: torch.Tensor, n_state: int, hparams: Config) -> torch.Tensor:
    nx = x.shape[-1]
    h_fc = gelu(conv1d(x, nf=n_state))
    h_proj = conv1d(h_fc, nf=nx)
    return h_proj


def block(
    x: torch.Tensor,
    past: torch.Tensor,
    hparams: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_state = x.shape[-1]

    # Normalize input tensor
    ln_1 = norm(x)

    # Attention and addition
    attend, present = attn(ln_1, n_state=n_state, past=past, hparams=hparams)
    x += attend

    # Re-normalize after adding attention
    ln_2 = norm(x)

    # MLP and addition
    mlp_output = mlp(ln_2, n_state * 4, hparams)
    x += mlp_output

    return x, present


def expand_tile(value: int | torch.Tensor, size: int) -> torch.Tensor:
    """Add a new axis of given size."""
    if isinstance(value, int):
        value = torch.Tensor([value])
    dims = len(list(value.shape))
    return torch.tile(torch.unsqueeze(value, dim=0), dims=[size] + ([1] * dims))


def positions_for(tokens: torch.Tensor, past_length: int) -> torch.Tensor:
    batch_size = tokens.shape[0]
    nsteps = tokens.shape[1]
    return expand_tile(past_length + torch.arange(nsteps), batch_size)


def gather(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather tensor along the specified dimension."""
    return torch.index_select(x, dim=0, index=indices)


def get_past(past: torch.Tensor, hparams: Config) -> torch.Tensor:
    # If past is not None, unpack the tensor along axis=1 and add dimensions
    if past is not None:
        past = torch.unbind(past, dim=1)
    # If no past is provided, create an empty list of size hparams.n_layer
    else:
        past = torch.zeros(hparams.n_layer, dtype=torch.float32)
    return past


def model(
    hparams: Config,
    X: torch.Tensor,
    past: torch.Tensor = None,
    reuse: bool = False,
):
    results = {}
    batch, sequence = list(X.shape)

    # Set positional encodings
    wpe = torch.nn.Embedding(hparams.n_ctx, hparams.n_embd)
    wpe.weight.normal_(mean=0.0, std=0.01)

    # Set word embeddings
    wte = torch.nn.Embedding(hparams.n_vocab, hparams.n_embd)
    wte.weight.normal_(mean=0.0, std=0.02)

    #
    past_length = 0 if past is None else past.shape[-2]
    # Hidden layer
    h = gather(wte, X) + gather(wpe, positions_for(X, past_length))

    # Transformer
    pasts = get_past(past, hparams)
    assert len(pasts) == hparams.n_layer

    #
    presents = []
    for layer, past in enumerate(pasts):
        h, present = block(h, past=past, hparams=hparams)
        presents.append(present)

    # NOTE: We need to stack, not squeeze. The tensors are concatenated.
    # Stacking is horizontal by default.
    results["present"] = torch.stack(presents, axis=1)
    h = norm(h)

    # Language model loss.  Do tokens <n predict token n?
    h_flat = torch.reshape(h, [batch * sequence, hparams.n_embd])
    logits = torch.matmul(h_flat, wte.T)
    logits = torch.reshape(logits, [batch, sequence, hparams.n_vocab])
    results["logits"] = logits
    return results


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, config: Config, std: float = 0.02):
        self.gelu = torch.nn.GELU()
        self.conv1d = torch.nn.Conv1d(config.n_embed, config.n_head)
        # Weight initialization
        torch.nn.init.normal_(self.conv1d.weight, std=std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        c_fc = self.gelu(self.conv1d(x))
        c_proj = self.conv1d(c_fc)
        return c_proj


class Block(torch.nn.Module):
    def __init__(self, config: Config):
        super(Block, self).__init__()
        self.config = config

        self.ln_1 = torch.nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = torch.nn.MultiheadAttention(
            config.n_embed, config.n_head, dropout=config.attn_pdrop
        )
        self.ln_2 = torch.nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MultiLayerPerceptron(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o, w = self.attn(self.ln_1(x), self.config.n_ctx)
        x = x + o
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, w


class GPT(torch.nn.Module):
    def __init__(self, config: Config):
        super(GPT, self).__init__(config)

        self.wpe = torch.nn.Embedding(config.n_positions, config.n_embd)
        self.wte = torch.nn.Embedding(config.vocab_size, config.n_embd)

        self.drop = torch.nn.Dropout(config.embd_pdrop)
        self.h = torch.nn.ModuleList(self.init_blocks())
        self.ln_f = torch.nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.apply(self.init_weights)

    def init_blocks(self) -> list[Block]:
        return [
            Block(self.config.n_ctx, self.config, scale=True)
            for _ in range(self.config.n_layer)
        ]

    def init_weights(self):
        with torch.no_grad():
            # Initialize position embeddings
            self.wpe.weight.normal_(std=0.1)

            # Initialize word embeddings
            self.wte.weight.normal_(mean=0, std=0.2)

            # Initialize attention and feedforward layers in transformer blocks
            for block in self.h:
                for layer in block._modules:
                    if isinstance(layer, (torch.nn.Linear, torch.nn.Conv1d)):
                        torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, input_ids):
        position_ids = torch.arange(
            0,
            input_ids.size(-1),
            dtype=torch.long,
            device=input_ids.device,
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(hidden_states)
        hidden_states = self.ln_f(hidden_states)

        return hidden_states
