#!/usr/bin/env python3
"""Latest official-tasks v1.1.2 Epoch-0 leaderboard candidates."""
import torch

try:
    from . import _C
except ImportError as exc:
    raise ImportError("Cannot import _C. Build and install the package first.") from exc

__all__ = ['gqa', 'grouped_matmul_swiglu_quant', 'gru', 'lstm', 'mha', 'mla', 'mla_prolog', 'sparse_flash_attention', 'adaptive_avg_pool_3d', 'add_rms_norm_dynamic_quant', 'conv_2d', 'conv_3d_backprop_filter', 'depthwise_conv_2d', 'dequant_swiglu_quant', 'dilation_2d', 'engram_gate_fusion', 'grouped_matmul', 'mhc_sinkhorn', 'moe_finalize_routing', 'moe_gating_top_k_softmax', 'moe_re_routing', 'quant_matmul', 'roi_align', 'strided_slice', 'top_k', 'transpose', 'unique', 'weight_quant_batch_matmul', 'apply_adam_w', 'apply_rotary_pos_emb', 'cross_entropy_loss', 'cummin', 'gather', 'gcd', 'grid_sampler_3d', 'maximum', 'resize_bilinear', 'rms_norm', 'scatter', 'softmax']

def gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaleValue: float = -1.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """Grouped query attention, y = softmax(Q K^T * scale) V per (batch, head)."""
    return torch.ops.cann_bench.gqa(query, key, value, scaleValue, is_causal)

def grouped_matmul_swiglu_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
):
    """Fused grouped matmul + dequant + SwiGLU + per-token int8 quant.

    Returns (y, y_scale): y is int8 [M, N/2], y_scale is float32 [M].
    """
    return torch.ops.cann_bench.grouped_matmul_swiglu_quant(
        x, weight, weight_scale, x_scale, group_list
    )

def gru(x: torch.Tensor, weight_ih, weight_hh, inputSize: int, hiddenSize: int,
        numLayers: int = 1, bias: bool = True, batchFirst: bool = False,
        dropout: float = 0.0, bidirectional: bool = False,
        bias_ih=None, bias_hh=None, h0=None):
    """Multi-layer (optionally bidirectional) GRU, matching torch.nn.GRU.

    Returns (y, hn).  See task/reference.py for the exact semantics.
    """
    return torch.ops.cann_bench.gru(x, list(weight_ih), list(weight_hh),
                                    inputSize, hiddenSize, numLayers, bias,
                                    batchFirst, dropout, bidirectional,
                                    None if bias_ih is None else list(bias_ih),
                                    None if bias_hh is None else list(bias_hh),
                                    h0)

def lstm(x: torch.Tensor, weight_ih, weight_hh, inputSize: int, hiddenSize: int,
         numLayers: int = 1, bias: bool = True, batchFirst: bool = False, dropout: float = 0.0,
         bidirectional: bool = False, projSize: int = 0, bias_ih=None, bias_hh=None,
         weight_hr=None, h0=None, c0=None):
    """Multi-layer / bidirectional / projected LSTM (torch.nn.LSTM semantics)."""
    return torch.ops.cann_bench.lstm(
        x, weight_ih, weight_hh, inputSize, hiddenSize, numLayers, bias, batchFirst,
        dropout, bidirectional, projSize, bias_ih, bias_hh, weight_hr, h0, c0)

def mha(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        scaleValue: float = -1.0, is_causal: bool = False) -> torch.Tensor:
    """Multi-head attention on already-split heads.

    y = softmax(query @ key^T * scaleValue) @ value, with right-bottom aligned
    causal masking when is_causal is True.
    """
    return torch.ops.cann_bench.mha(query, key, value, scaleValue, is_causal)

def mla(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
    numKVHeads: int = 1,
    scaleValue: float = -1.0,
    inputLayout: str = "BSND",
    is_causal: bool = False,
) -> torch.Tensor:
    """Multi-head latent attention (attention part only).

    Q = concat(Q_nope, Q_rope), K = concat(K_nope, K_rope), V = v
    y = softmax(Q @ K^T * scaleValue) @ V
    """
    return torch.ops.cann_bench.mla(
        q_nope, q_rope, k_nope, k_rope, v,
        numKVHeads, scaleValue, inputLayout, is_causal,
    )

def mla_prolog(
    token_x: torch.Tensor,
    w_dq: torch.Tensor,
    w_uq_qr: torch.Tensor,
    w_uk: torch.Tensor,
    w_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    n_heads: int,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
):
    """Multi-head latent attention prolog: query/key projection + RMSNorm + RoPE.

    Returns ``(query, query_rope, c_kv, k_rope)``.
    """
    return torch.ops.cann_bench.mla_prolog(
        token_x,
        w_dq,
        w_uq_qr,
        w_uk,
        w_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        rope_sin,
        rope_cos,
        n_heads,
        rmsnorm_epsilon_cq,
        rmsnorm_epsilon_ckv,
    )

def sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparseIndices: torch.Tensor,
    scaleValue: float,
    inputLayout: str = "BSND",
    is_causal: bool = False,
) -> torch.Tensor:
    """Sparse attention over the KV positions selected by ``sparseIndices`` (GQA aware)."""
    return torch.ops.cann_bench.sparse_flash_attention(
        query, key, value, sparseIndices, float(scaleValue), str(inputLayout), bool(is_causal)
    )

def adaptive_avg_pool_3d(x: torch.Tensor, output_size) -> torch.Tensor:
    """3D adaptive average pooling.

    Args:
        x: (N, C, D, H, W) tensor.
        output_size: sequence of three positive ints (output_d, output_h, output_w).

    Returns:
        (N, C, output_d, output_h, output_w) tensor with the same dtype as x.
    """
    return torch.ops.cann_bench.adaptive_avg_pool_3d(x, list(output_size))

def add_rms_norm_dynamic_quant(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
):
    """Add + RMSNorm + per-token symmetric dynamic quantization.

    Returns (y_int8, xOut, scaleOut_fp32).
    """
    return torch.ops.cann_bench.add_rms_norm_dynamic_quant(x1, x2, gamma, epsilon)

def conv_2d(x: torch.Tensor, filter: torch.Tensor, bias: torch.Tensor,
            strides, pads, dilations=None) -> torch.Tensor:
    """y = CONV(x, filter) + bias"""
    if dilations is None:
        dilations = [1, 1]
    if len(dilations) == 1:
        dilations = [dilations[0], dilations[0]]
    if len(strides) == 1:
        strides = [strides[0], strides[0]]
    return torch.ops.cann_bench.conv_2d(x, filter, bias, list(strides), list(pads), list(dilations))

def conv_3d_backprop_filter(x: torch.Tensor, grad: torch.Tensor, strides, pads, dilations,
                            groups: int = 1, filter_size=None) -> torch.Tensor:
    """Gradient of a 3D convolution with respect to the filter.

    ``y[co, ci, kd, kh, kw]`` is the filter gradient for input ``x`` and output
    gradient ``grad`` using ``strides``, ``pads`` (6-element, the front/top/left
    entries are used), ``dilations``, ``groups`` and ``filter_size``.
    """
    if filter_size is None:
        raise ValueError("filter_size is required (proto.yaml declares it as a required attr)")
    return torch.ops.cann_bench.conv_3d_backprop_filter(
        x, grad, list(strides), list(pads), list(dilations), int(groups), list(filter_size)
    )

def depthwise_conv_2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, kernelSize, stride,
                      padding, dilation, groups: int) -> torch.Tensor:
    """Depthwise 2D convolution (groups == C): y[n,c,h,w] = bias[c] +
    sum_{kh,kw} x[n,c,h*sh+kh*dh-ph, w*sw+kw*dw-pw] * weight[c,kh,kw]."""
    return torch.ops.cann_bench.depthwise_conv_2d(x, weight, bias, kernelSize, stride, padding,
                                                 dilation, groups)

def dequant_swiglu_quant(x: torch.Tensor, weight_scale=None, activation_scale=None,
                         quant_scale=None, activate_left: bool = False):
    """Dequant + SwiGLU + per-token dynamic int8 quant.

    Returns (y, scale): y is int8 with the last dim halved, scale is float32 with
    shape x.shape[:-1].
    """
    return torch.ops.cann_bench.dequant_swiglu_quant(
        x, weight_scale, activation_scale, quant_scale, activate_left)

def dilation_2d(x: torch.Tensor, filter: torch.Tensor, strides: list,
                rates: list, padding_mode: str = "SAME", pads: list = [0, 0, 0, 0],
                ceil_mode: bool = False, data_format: str = "NHWC") -> torch.Tensor:
    """NHWC 2D morphological dilation with an additive structuring element."""
    return torch.ops.cann_bench.dilation_2d(
        x, filter, strides, rates, padding_mode, pads, ceil_mode, data_format)

def engram_gate_fusion(
    keys: torch.Tensor,
    hidden_states: torch.Tensor,
    value: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    conv_norm_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state=None,
    hc_mult: int = 4,
    hidden_size: int = 1024,
    kernel_size: int = 4,
    dilation: int = 3,
    norm_eps: float = 1e-5,
):
    """Dual RMSNorm gate + broadcast multiply + ShortConv + residual (DeepSeek Engram).

    Returns (output, conv_state_out).
    """
    return torch.ops.cann_bench.engram_gate_fusion(
        keys, hidden_states, value,
        norm1_weight, norm2_weight, conv_norm_weight, conv_weight,
        conv_state, hc_mult, hidden_size, kernel_size, dilation, norm_eps)

def grouped_matmul(x: torch.Tensor, weight: torch.Tensor, bias=None, group_list=None,
                   split_item: int = 0, transpose_weight: bool = False):
    """Grouped matmul: y[rows_g] = x[rows_g] @ weight[g] (+ bias[g]).

    split_item 0/1 -> list of E tensors [m_i, N]; split_item 2/3 -> [ [M, N] ].
    """
    return torch.ops.cann_bench.grouped_matmul(x, weight, bias, group_list, split_item, transpose_weight)

def mhc_sinkhorn(comb: torch.Tensor, iter_step: int = 20, eps: float = 1e-6) -> torch.Tensor:
    return torch.ops.cann_bench.mhc_sinkhorn(comb, iter_step, eps)

def moe_finalize_routing(
    expanded_permuted_rows: torch.Tensor,
    expanded_src_to_dst_row: torch.Tensor,
    skip1: torch.Tensor = None,
    skip2: torch.Tensor = None,
    bias: torch.Tensor = None,
    scales: torch.Tensor = None,
    expert_for_source_row: torch.Tensor = None,
    drop_pad_mode: int = 0,
) -> torch.Tensor:
    """MoE finalize routing: merge MoE FFN expert outputs with residual and bias."""
    return torch.ops.cann_bench.moe_finalize_routing(
        expanded_permuted_rows,
        expanded_src_to_dst_row,
        skip1,
        skip2,
        bias,
        scales,
        expert_for_source_row,
        drop_pad_mode,
    )

def moe_gating_top_k_softmax(
    x: torch.Tensor,
    finished: torch.Tensor = None,
    k: int = 1,
):
    """Fused MoE gating: softmax over the last dim followed by top-k.

    Returns (y, expert_idx, row_idx); y holds the k largest softmax values in
    descending order, expert_idx the matching expert ids (num_experts sentinel for
    finished rows) and row_idx the flattened global position indices.
    """
    if finished is None:
        return torch.ops.cann_bench.moe_gating_top_k_softmax(x, None, k)
    return torch.ops.cann_bench.moe_gating_top_k_softmax(x, finished, k)

def moe_re_routing(
    tokens: torch.Tensor,
    expert_token_num_per_rank: torch.Tensor,
    per_token_scales: torch.Tensor = None,
    expert_token_num_type: int = 1,
    idx_type: int = 0,
):
    """MoE re-routing: permute tokens (and their scales) into expert-major order.

    Returns (permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num).
    """
    return torch.ops.cann_bench.moe_re_routing(
        tokens,
        expert_token_num_per_rank,
        per_token_scales,
        expert_token_num_type,
        idx_type,
    )

def quant_matmul(x1: torch.Tensor, x2: torch.Tensor, scale: torch.Tensor,
                 offset=None, pertoken_scale=None, bias=None, output_dtype=None) -> torch.Tensor:
    """int8 x int8 quantized matmul with dequantization.

    out[..., m, n] = (x1 @ x2 + bias_int32) * scale + offset) * pertoken_scale
                     + bias_float, cast to float16/bfloat16.
    """
    return torch.ops.cann_bench.quant_matmul(
        x1, x2, scale, offset, pertoken_scale, bias, output_dtype)

def roi_align(
    x: torch.Tensor,
    boxes: torch.Tensor,
    outputHeight: int,
    outputWidth: int,
    spatial_scale: float,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> torch.Tensor:
    """Region-of-interest pooling for feature maps of non-uniform input size.

    Formula: y = roi_align(x, boxes, output_size)

    Args:
        x: input feature map [B, C, H, W]
        boxes: ROI boxes [N, 5], each row (batch_idx, x0, y0, x1, y1)
        outputHeight: output height
        outputWidth: output width
        spatial_scale: spatial scale factor
        sampling_ratio: sampling ratio (-1 or 0 -> auto)
        aligned: whether to align

    Returns:
        output tensor [N, C, outputHeight, outputWidth]
    """
    return torch.ops.cann_bench.roi_align(
        x, boxes, outputHeight, outputWidth, spatial_scale, sampling_ratio, aligned
    )

def strided_slice(x: torch.Tensor, begin: list, end: list, strides: list,
                  begin_mask: int = 0, end_mask: int = 0, ellipsis_mask: int = 0,
                  shrink_axis_mask: int = 0, new_axis_mask: int = 0) -> torch.Tensor:
    """Strided (multi-dimensional) slice of ``x`` with mask attributes."""
    return torch.ops.cann_bench.strided_slice(
        x, begin, end, strides, begin_mask, end_mask, ellipsis_mask,
        shrink_axis_mask, new_axis_mask)

def top_k(
    x: torch.Tensor, k: int, dim: int, largest: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the k largest (or smallest) elements of x along dim, with their indices."""
    return torch.ops.cann_bench.top_k(x, k, dim, largest)

def transpose(x: torch.Tensor, perm: list) -> torch.Tensor:
    """Permute the dimensions of x and materialise a contiguous result."""
    return torch.ops.cann_bench.transpose(x, perm)

def unique(x: torch.Tensor, return_inverse: bool = False):
    """Remove duplicate elements of x.

    Returns the sorted unique values ``y`` (same dtype as x, 1-D).  When
    ``return_inverse`` is true the inverse index ``inverse`` (int64, shaped like
    ``x``) is returned as well, so that ``x.flatten() == y[inverse.flatten()]``.
    """
    res = torch.ops.cann_bench.unique(x, return_inverse)
    if return_inverse:
        return res[0], res[1]
    return res[0]

def weight_quant_batch_matmul(x: torch.Tensor, weight: torch.Tensor,
                              antiquantScale: torch.Tensor, antiquantOffset=None,
                              bias=None) -> torch.Tensor:
    """y = x @ ((weight + antiquantOffset) * antiquantScale) + bias"""
    return torch.ops.cann_bench.weight_quant_batch_matmul(
        x, weight, antiquantScale, antiquantOffset, bias)

def apply_adam_w(var: torch.Tensor, grad: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
                 lr: float, beta1: float, beta2: float, weight_decay: float,
                 epsilon: float = 1e-8, step: int = 1, maximize: bool = False) -> torch.Tensor:
    """AdamW optimizer (decoupled weight decay): y = var -/+ lr * (m_hat/(sqrt(v_hat)+eps) + wd*var)."""
    return torch.ops.cann_bench.apply_adam_w(var, grad, m, v, lr, beta1, beta2, weight_decay,
                                             epsilon, step, maximize)

def apply_rotary_pos_emb(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor,
                         sin: torch.Tensor, layout: int = 0, rotaryMode: str = "half"):
    """Rotary position embedding applied to query and key.

    y = x * cos_full + rotate_half(x) * sin_full
    """
    return torch.ops.cann_bench.apply_rotary_pos_emb(
        query, key, cos, sin, layout, rotaryMode)

def cross_entropy_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> torch.Tensor:
    """CrossEntropyLoss, class axis = dim 1 (channel first)."""
    return torch.ops.cann_bench.cross_entropy_loss(input, target, reduction, ignore_index)

def cummin(input: torch.Tensor, dim: int = -1):
    return torch.ops.cann_bench.cummin(input, dim)

def gather(x: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """torch.gather semantics: y.shape == index.shape, y.dtype == x.dtype."""
    return torch.ops.cann_bench.gather(x, index, dim)

def gcd(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.ops.cann_bench.gcd(x1, x2)

def grid_sampler_3d(x: torch.Tensor, grid: torch.Tensor, interpolation_mode: str = "bilinear",
                    padding_mode: str = "zeros", align_corners: bool = False) -> torch.Tensor:
    return torch.ops.cann_bench.grid_sampler_3d(x, grid, interpolation_mode, padding_mode, align_corners)

def maximum(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Elementwise maximum with PyTorch broadcasting: y = max(x1, x2)."""
    return torch.ops.cann_bench.maximum(x1, x2)

def resize_bilinear(x: torch.Tensor, output_size=None, align_corners: bool = False, scale_factor=None) -> torch.Tensor:
    return torch.ops.cann_bench.resize_bilinear(x, output_size, align_corners, scale_factor)

def rms_norm(x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    return torch.ops.cann_bench.rms_norm(x, gamma, epsilon)

def scatter(data: torch.Tensor, dim: int, indices: torch.Tensor, updates: torch.Tensor,
            reduce: str = None) -> torch.Tensor:
    """Scatter updates into data along dim, with an optional commutative reduce."""
    return torch.ops.cann_bench.scatter(data, dim, indices, updates, reduce)

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax normalization along dim."""
    return torch.ops.cann_bench.softmax(x, dim)
