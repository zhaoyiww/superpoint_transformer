import torch
from torch import nn
from torch_scatter import scatter_sum
from torch_geometric.utils import softmax
from src.utils.nn import build_qk_scale_func


__all__ = ['SelfAttentionBlock']


class SelfAttentionBlock(nn.Module):
    """SelfAttentionBlock is intended to be used in a residual fashion
    (or not) in TransformerBlock.

    Inspired by: https://github.com/microsoft/Swin-Transformer

    :param dim: int
        Dimension of the features space on which the attention block
        operates
    :param num_heads: int
        Number of attention heads
    :param in_dim: int
        Dimension of the input features. If specified, the features will
        be mapped from `in_dim` to `dim` with a linear projection
    :param out_dim: int
        Dimension of the output features. If specified, the features
        will be mapped from `dim` to `out_dim` with a linear projection
    :param qkv_bias: bool
        Whether the linear layers producing queries, keys, and
        values should have a bias
    :param qk_dim: int
        Dimension of the queries and keys
    :param qk_scale: str
        Scaling applied to the query*key product before the softmax.
        More specifically, one may want to normalize the query-key
        compatibilities based on the number of dimensions (referred
        to as 'd' here) as in a vanilla Transformer implementation,
        or based on the number of neighbors each node has in the
        attention graph (referred to as 'g' here). If nothing is
        specified the scaling will be `1 / (sqrt(d) * sqrt(g))`,
        which is equivalent to passing `'d.g'`. Passing `'d+g'` will
        yield `1 / (sqrt(d) + sqrt(g))`. Meanwhile, passing 'd' will
        yield `1 / sqrt(d)`, and passing `'g'` will yield
        `1 / sqrt(g)`
    :param attn_drop: float
        Dropout on the attention weights
    :param drop: float
        Dropout on the output features
    :param in_rpe_dim: int
        Dimension of the features passed as input for relative
        positional encoding computation (i.e. edge features)
    :param k_rpe: bool
        Whether keys should receive relative positional encodings
        computed from edge features
    :param q_rpe: bool
        Whether queries should receive relative positional encodings
        computed from edge features
    :param v_rpe: bool
        Whether values should receive relative positional encodings
        computed from edge features
    :param k_delta_rpe: bool
        Whether keys should receive relative positional encodings
        computed from the difference between source and target node
        features
    :param q_delta_rpe: bool
        Whether queries should receive relative positional encodings
        computed from the difference between source and target node
        features
    :param qk_share_rpe: bool
        Whether queries and keys should use the same parameters for
        building relative positional encodings
    :param q_on_minus_rpe: bool
        Whether relative positional encodings for queries should be
        computed on the opposite of features used for keys. This allows,
        for instance, to break the symmetry when `qk_share_rpe` but we
        want relative positional encodings to capture different meanings
        for keys and queries
    :param heads_share_rpe: bool
        whether attention heads should share the same parameters for
        building relative positional encodings
    """

    def __init__(
            self,
            dim,
            num_heads=1,
            in_dim=None,
            out_dim=None,
            qkv_bias=True,
            qk_dim=8,
            qk_scale=None,
            attn_drop=None,
            drop=None,
            in_rpe_dim=18,
            k_rpe=False,
            q_rpe=False,
            v_rpe=False,
            k_delta_rpe=False,
            q_delta_rpe=False,
            qk_share_rpe=False,
            q_on_minus_rpe=False,
            heads_share_rpe=False):
        super().__init__()

        assert dim % num_heads == 0, f"dim must be a multiple of num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.qk_scale = build_qk_scale_func(dim, num_heads, qk_scale)
        self.heads_share_rpe = heads_share_rpe

        self.qkv = nn.Linear(dim, qk_dim * 2 * num_heads + dim, bias=qkv_bias)  # TODO: only 1 value for all heads ?

        # TODO: define relative positional encoding parameters and
        #  truncated-normal initialize them, see Swin-T implementation:
        #  https://github.com/microsoft/Swin-Transformer/blob/e43ac64ce8abfe133ae582741ccaf6761eea05f7/models/swin_transformer.py#L122

        # TODO: k/q/v RPE, pos/edge attr/both RPE, MLP/vector attention,
        #  mlp on pos/learnable lookup table/FFN/learnable FFN...

        # Build the RPE encoders, with the option of sharing weights
        # across all heads
        qk_rpe_dim = qk_dim if heads_share_rpe else qk_dim * num_heads
        v_rpe_dim = dim // num_heads if heads_share_rpe else dim

        if not isinstance(k_rpe, bool):
            self.k_rpe = k_rpe
        else:
            self.k_rpe = nn.Linear(in_rpe_dim, qk_rpe_dim) if k_rpe else None

        if not isinstance(q_rpe, bool):
            self.q_rpe = q_rpe
        else:
            self.q_rpe = nn.Linear(in_rpe_dim, qk_rpe_dim) if \
                q_rpe and not (k_rpe and qk_share_rpe) else None

        if not isinstance(k_delta_rpe, bool):
            self.k_delta_rpe = k_delta_rpe
        else:
            self.k_delta_rpe = nn.Linear(dim, qk_rpe_dim) if k_delta_rpe \
                else None

        if not isinstance(q_delta_rpe, bool):
            self.q_delta_rpe = q_delta_rpe
        else:
            self.q_delta_rpe = nn.Linear(dim, qk_rpe_dim) if \
                q_delta_rpe and not (k_delta_rpe and qk_share_rpe) \
                else None

        self.qk_share_rpe = qk_share_rpe
        self.q_on_minus_rpe = q_on_minus_rpe

        if not isinstance(v_rpe, bool):
            self.v_rpe = v_rpe
        else:
            self.v_rpe = nn.Linear(in_rpe_dim, v_rpe_dim) if v_rpe else None

        self.in_proj = nn.Linear(in_dim, dim) if in_dim is not None else None
        self.out_proj = nn.Linear(dim, out_dim) if out_dim is not None else None

        self.attn_drop = nn.Dropout(attn_drop) \
            if attn_drop is not None and attn_drop > 0 else None
        self.out_drop = nn.Dropout(drop) \
            if drop is not None and drop > 0 else None

    def forward(self, x, edge_index, edge_attr=None):
        """
        :param x: Tensor of shape (N, Cx)
            Node features
        :param edge_index: LongTensor of shape (2, E)
            Source and target indices for the edges of the attention
            graph. Source indicates the querying element, while Target
            indicates the key elements
        :param edge_attr: FloatTensor or shape (E, Ce)
            Edge attributes for relative pose encoding
        :return:
        """
        N = x.shape[0]
        E = edge_index.shape[1]
        H = self.num_heads
        D = self.qk_dim
        DH = D * H

        # Optional linear projection of features
        if self.in_proj is not None:
            x = self.in_proj(x)

        # Compute queries, keys and values
        # qkv = self.qkv(x).view(N, 3, self.num_heads, self.dim // self.num_heads)
        qkv = self.qkv(x)

        # # Separate and expand queries, keys, values and indices to edge
        # # shape
        # s = edge_index[0]  # [E]
        # t = edge_index[1]  # [E]
        # q = qkv[s, 0]  # [E, H, C // H]
        # k = qkv[t, 1]  # [E, H, C // H]
        # v = qkv[t, 2]  # [E, H, C // H]

        # Separate queries, keys, values
        q = qkv[:, :DH].view(N, H, D)        # [N, H, D]
        k = qkv[:, DH:2 * DH].view(N, H, D)  # [N, H, D]
        v = qkv[:, 2 * DH:].view(N, H, -1)   # [N, H, C // H]

        # Expand queries, keys and values to edges
        s = edge_index[0]  # [E]
        t = edge_index[1]  # [E]
        q = q[s]  # [E, H, D]
        k = k[t]  # [E, H, D]
        v = v[t]  # [E, H, C // H]

        # Apply scaling on the queries.
        q = q * self.qk_scale(s)

        # TODO: add the relative positional encodings to the
        #  compatibilities here
        #  - k_rpe, q_rpe, v_rpe
        #  - pos difference, absolute distance, squared distance, centroid distance, edge distance, ...
        #  - with/out edge attributes
        #  - mlp (L-LN-A-L), learnable lookup table (see Stratified Transformer)
        #  - scalar rpe, vector rpe (see Stratified Transformer)

        # Relative positional encoding from edge features for keys
        if self.k_rpe is not None and edge_attr is not None:
            rpe = self.k_rpe(edge_attr)

            # Expand RPE to all heads if heads share the RPE encoder
            if self.heads_share_rpe:
                rpe = rpe.repeat(1, H)

            k = k + rpe.view(E, H, -1)

        # Relative positional encoding from edge features for queries
        if self.q_rpe is not None and edge_attr is not None:
            if self.q_on_minus_rpe:
                rpe = self.q_rpe(-edge_attr)
            else:
                rpe = self.q_rpe(edge_attr)

            # Expand RPE to all heads if heads share the RPE encoder
            if self.heads_share_rpe:
                rpe = rpe.repeat(1, H)

            q = q + rpe.view(E, H, -1)
        elif self.k_rpe is not None and self.qk_share_rpe and edge_attr is not None:
            if self.q_on_minus_rpe:
                rpe = self.k_rpe(-edge_attr)
            else:
                rpe = self.k_rpe(edge_attr)

            # Expand RPE to all heads if heads share the RPE encoder
            if self.heads_share_rpe:
                rpe = rpe.repeat(1, H)

            q = q + rpe.view(E, H, -1)

        # Relative positional encoding from node delta features for keys
        if self.k_delta_rpe is not None:
            rpe = self.k_delta_rpe(x[edge_index[1]] - x[edge_index[0]])

            # Expand RPE to all heads if heads share the RPE encoder
            if self.heads_share_rpe:
                rpe = rpe.repeat(1, H)

            k = k + rpe.view(E, H, -1)

        # Relative positional encoding from node delta features for
        # queries
        if self.q_delta_rpe is not None:
            if self.q_on_minus_rpe:
                rpe = self.q_delta_rpe(x[edge_index[0]] - x[edge_index[1]])
            else:
                rpe = self.q_delta_rpe(x[edge_index[1]] - x[edge_index[0]])

            # Expand RPE to all heads if heads share the RPE encoder
            if self.heads_share_rpe:
                rpe = rpe.repeat(1, H)

            q = q + rpe.view(E, H, -1)
        elif self.k_delta_rpe is not None and self.qk_share_rpe and edge_attr is not None:
            if self.q_on_minus_rpe:
                rpe = self.k_delta_rpe(x[edge_index[0]] - x[edge_index[1]])
            else:
                rpe = self.k_delta_rpe(x[edge_index[1]] - x[edge_index[0]])

            # Expand RPE to all heads if heads share the RPE encoder
            if self.heads_share_rpe:
                rpe = rpe.repeat(1, H)

            q = q + rpe.view(E, H, -1)

        # Relative positional encoding from edge features for values
        if self.v_rpe is not None and edge_attr is not None:
            rpe = self.v_rpe(edge_attr)

            # Expand RPE to all heads if heads share the RPE encoder
            if self.heads_share_rpe:
                rpe = rpe.repeat(1, H)

            v = v + rpe.view(E, H, -1)

        # Compute compatibility scores from the query-key products
        compat = torch.einsum('ehd, ehd -> eh', q, k)  # [E, H]

        # Compute the attention scores with scaled softmax
        attn = softmax(compat, index=s, dim=0, num_nodes=N)  # [E, H]

        # Optional attention dropout
        if self.attn_drop is not None:
            attn = self.attn_drop(attn)

        # Apply the attention on the values
        x = (v * attn.unsqueeze(-1)).view(E, self.dim)  # [E, C]
        x = scatter_sum(x, s, dim=0, dim_size=N)  # [N, C]

        # Optional linear projection of features
        if self.out_proj is not None:
            x = self.out_proj(x)  # [N, out_dim]

        # Optional dropout on projection of features
        if self.out_drop is not None:
            x = self.out_drop(x)  # [N, C or out_dim]

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'
