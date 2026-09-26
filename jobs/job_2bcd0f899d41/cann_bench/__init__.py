#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Best-of-trajectory CANNBench exports assembled from canonical G1-G4 evidence."""
__version__ = "1.0.0"
import torch
from typing import List, Optional
from typing import List

try:
    from . import _C
except ImportError as e:
    raise ImportError(
        "Cannot import _C. Please make sure the `cann_bench` package is properly installed. "
    ) from e

__all__ = ['adaptive_avg_pool_3d', 'add_rms_norm_dynamic_quant', 'conv_2d', 'conv_3d_backprop_filter', 'depthwise_conv_2d', 'dequant_swiglu_quant', 'dilation_2d', 'engram_gate_fusion', 'grouped_matmul', 'mhc_sinkhorn', 'moe_finalize_routing', 'moe_gating_top_k_softmax', 'moe_re_routing', 'nms', 'quant_matmul', 'roi_align', 'strided_slice', 'top_k', 'transpose', 'unique', 'weight_quant_batch_matmul', 'gqa', 'grouped_matmul_swiglu_quant', 'gru', 'lstm', 'mha', 'mla', 'mla_prolog', 'sparse_flash_attention', 'gcd', 'resize_bilinear', 'arg_max', 'cummin', 'scatter', 'grid_sampler_3d', 'apply_adam_w', 'apply_rotary_pos_emb', 'cross_entropy_loss', 'gather', 'rms_norm', 'softmax', 'maximum']

def _as_tensor_list(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        out = [t for t in v if t is not None]
        return out if out else None
    return [v]

def _gru_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _gru_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    return None

def _gru_tensor_list(value):
    """Normalise an optional TensorList: None / [None] / () all mean "absent"."""
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [value]
    try:
        items = list(value)
    except TypeError:
        return []
    return [t for t in items if t is not None]

def adaptive_avg_pool_3d(x: torch.Tensor, output_size) -> torch.Tensor:
    """3D adaptive average pooling.

    Args:
        x: [N, C, D, H, W] input tensor.
        output_size: [output_d, output_h, output_w].
    Returns:
        [N, C, output_d, output_h, output_w] tensor, same dtype as x.
    """
    if not isinstance(output_size, (list, tuple)):
        output_size = list(output_size)
    return torch.ops.cann_bench.adaptive_avg_pool_3d(x, list(output_size))

def add_rms_norm_dynamic_quant(
    x1: torch.Tensor, x2: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6
):
    """y, xOut, scaleOut = quantize(rmsnorm(x1 + x2) * gamma)"""
    return torch.ops.cann_bench.add_rms_norm_dynamic_quant(x1, x2, gamma, epsilon)

def conv_2d(x: torch.Tensor, filter: torch.Tensor, bias: torch.Tensor,
            strides=None, pads=None, dilations=None) -> torch.Tensor:
    """y = conv2d(x, filter) + bias with groups == 1."""
    if strides is None:
        strides = [1, 1]
    if pads is None:
        pads = [0, 0, 0, 0]
    if dilations is None:
        dilations = [1, 1]
    return torch.ops.cann_bench.conv_2d(x, filter, bias, strides, pads, dilations)

def conv_3d_backprop_filter(x: torch.Tensor, grad: torch.Tensor, strides=None, pads=None,
                            dilations=None, groups: int = 1, filter_size=None) -> torch.Tensor:
    """Conv3D filter gradient: y = conv3d_filter_grad(x, grad, filter_size)."""
    if strides is None:
        strides = [1, 1, 1]
    if pads is None:
        pads = [0, 0, 0, 0, 0, 0]
    if dilations is None:
        dilations = [1, 1, 1]
    if filter_size is None:
        raise ValueError("conv_3d_backprop_filter: filter_size is required")
    return torch.ops.cann_bench.conv_3d_backprop_filter(
        x, grad, list(strides), list(pads), list(dilations), int(groups), list(filter_size))

def depthwise_conv_2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    kernelSize,
    stride,
    padding,
    dilation,
    groups: int,
) -> torch.Tensor:
    """y[n,c,ho,wo] = bias[c] + sum_{kh,kw} x[n,c,ho*sh+kh*dh-ph, wo*sw+kw*dw-pw] * weight[c,kh,kw]."""
    return torch.ops.cann_bench.depthwise_conv_2d(
        x,
        weight,
        bias,
        list(kernelSize),
        list(stride),
        list(padding),
        list(dilation),
        int(groups),
    )

def dequant_swiglu_quant(x: torch.Tensor,
                         weight_scale=None,
                         activation_scale=None,
                         quant_scale=None,
                         activate_left: bool = False):
    """Fused dequant -> SwiGLU -> per-token dynamic quant.

    Returns (y, scale): y is [.., H] int8 and scale is [..] float32.
    """
    return torch.ops.cann_bench.dequant_swiglu_quant(
        x, weight_scale, activation_scale, quant_scale, activate_left)

def dilation_2d(
    x: torch.Tensor,
    filter: torch.Tensor,
    strides: List[int],
    rates: List[int],
    padding_mode: str = 'SAME',
    pads: Optional[List[int]] = None,
    ceil_mode: bool = False,
    data_format: str = 'NHWC',
) -> torch.Tensor:
    """2D morphological dilation with a per-channel structuring element.

    y[n, oh, ow, c] = max_{dy,dx} (x[n, oh*sh + rh*dy - pad_top, ow*sw + rw*dx - pad_left, c]
                                   + filter[dy, dx, c])
    """
    if pads is None:
        pads = [0, 0, 0, 0]
    return torch.ops.cann_bench.dilation_2d(
        x, filter, list(strides), list(rates), padding_mode, list(pads),
        bool(ceil_mode), data_format,
    )

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
    """EngramGateFusion: dual RMSNorm gate, broadcast multiply, ShortConv and residual."""
    if conv_state is not None and not isinstance(conv_state, torch.Tensor):
        conv_state = None
    return torch.ops.cann_bench.engram_gate_fusion(
        keys,
        hidden_states,
        value,
        norm1_weight,
        norm2_weight,
        conv_norm_weight,
        conv_weight,
        conv_state,
        int(hc_mult),
        int(hidden_size),
        int(kernel_size),
        int(dilation),
        float(norm_eps),
    )

def grouped_matmul(x, weight, bias=None, group_list=None, split_item: int = 0,
                   transpose_weight: bool = False):
    """Grouped matmul: y[rows_g] = x[rows_g] @ weight[g] (+ bias[g]).

    Returns E tensors (one per expert) when split_item is 0 or 1, and a single
    [M, N] tensor otherwise. The device side always materialises the full result,
    so the per-expert split is a pure metadata view.
    """
    return torch.ops.cann_bench.grouped_matmul(x, weight, bias, list(group_list),
                                               int(split_item), bool(transpose_weight))

def mhc_sinkhorn(comb: torch.Tensor, iter_step: int = 20, eps: float = 1e-6) -> torch.Tensor:
    """Linear-domain Sinkhorn-Knopp doubly-stochastic projection of [B, hc, hc]."""
    return torch.ops.cann_bench.mhc_sinkhorn(comb, iter_step, eps)

def moe_finalize_routing(
    expanded_permuted_rows: torch.Tensor,
    expanded_src_to_dst_row: torch.Tensor,
    skip1=None,
    skip2=None,
    bias=None,
    scales=None,
    expert_for_source_row=None,
    drop_pad_mode: int = 0,
) -> torch.Tensor:
    """Merge MoE FFN expert outputs with the shared-expert residual streams and expert bias.

    out(i, j) = skip1(i, j) + skip2(i, j)
              + sum_k scales(i, k) * ( expanded_permuted_rows[idx(i, k), j]
                                     (+ bias[expert_for_source_row(i, k), j]) )
    """
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

def moe_gating_top_k_softmax(x: torch.Tensor, finished=None, k: int = 1):
    """softmax(x, -1) fused with top-k.

    Returns (y, expert_idx, row_idx) with y = topk(softmax(x, -1), k) values,
    expert_idx the corresponding expert indices (num_expert sentinel on finished
    rows) and row_idx[r][j] = j * num_rows + r.
    """
    return torch.ops.cann_bench.moe_gating_top_k_softmax(x, finished, k)

def moe_re_routing(
    tokens: torch.Tensor,
    expert_token_num_per_rank: torch.Tensor,
    per_token_scales=None,
    expert_token_num_type: int = 1,
    idx_type: int = 0,
):
    """Reorder MoE tokens by expert.

    Args:
        tokens: (A, H) tokens to be re-laid out.
        expert_token_num_per_rank: (N, E) token counts per (rank, expert) cell,
            elements > 0 and sum == A.
        per_token_scales: optional (A,) float32 scales reordered alongside tokens.
        expert_token_num_type: 1 = count mode (only supported value).
        idx_type: 0 = gather index (only supported value).

    Returns:
        (permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num)
    """
    return torch.ops.cann_bench.moe_re_routing(
        tokens,
        expert_token_num_per_rank,
        per_token_scales,
        expert_token_num_type,
        idx_type,
    )

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """keep_indices = nms(boxes, scores, iou_threshold); output dtype is int64."""
    return torch.ops.cann_bench.nms(boxes, scores, iou_threshold)

def quant_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    pertoken_scale: torch.Tensor = None,
    bias: torch.Tensor = None,
    output_dtype: str = None,
    group_sizes=None,
) -> torch.Tensor:
    """Quantized matmul: out = dequant(x1 @ x2) with int8 inputs and fp16/bf16 output."""
    return torch.ops.cann_bench.quant_matmul(
        x1,
        x2,
        scale,
        pertoken_scale=pertoken_scale,
        bias=bias,
        output_dtype=output_dtype,
        group_sizes=group_sizes,
    )

def roi_align(
    x: torch.Tensor,
    boxes: torch.Tensor,
    outputHeight: int,
    outputWidth: int,
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> torch.Tensor:
    """ROIAlign: y = roi_align(x, boxes, output_size).

    Args:
        x: feature map [B, C, H, W].
        boxes: [numBoxes, 5], each row (batch_idx, x0, y0, x1, y1).
        outputHeight / outputWidth: pooled output size.
        spatial_scale: box coordinate -> feature map scale.
        sampling_ratio: -1 or 0 means "auto" (ceil(roi / output)).
        aligned: apply the -0.5 pixel shift when true.
    Returns:
        [numBoxes, C, outputHeight, outputWidth] with the dtype of x.
    """
    return torch.ops.cann_bench.roi_align(
        x,
        boxes,
        int(outputHeight),
        int(outputWidth),
        float(spatial_scale),
        int(sampling_ratio),
        bool(aligned),
    )

def strided_slice(
    x: torch.Tensor,
    begin: List[int],
    end: List[int],
    strides: List[int],
    begin_mask: int = 0,
    end_mask: int = 0,
    ellipsis_mask: int = 0,
    shrink_axis_mask: int = 0,
    new_axis_mask: int = 0,
) -> torch.Tensor:
    """TensorFlow style strided slice: x[begin[i]:end[i]:strides[i], ...] with masks."""
    return torch.ops.cann_bench.strided_slice(
        x,
        list(begin),
        list(end),
        list(strides),
        int(begin_mask),
        int(end_mask),
        int(ellipsis_mask),
        int(shrink_axis_mask),
        int(new_axis_mask),
    )

def top_k(x: torch.Tensor, k: int, dim: int, largest: bool = True):
    """y, idx = topk(x, k, dim, largest); idx dtype is int64."""
    return torch.ops.cann_bench.top_k(x, k, dim, largest)

def transpose(x: torch.Tensor, perm) -> torch.Tensor:
    """Permute the dimensions of ``x`` according to ``perm``."""
    return torch.ops.cann_bench.transpose(x, perm)

def unique(x: torch.Tensor, return_inverse: bool = False):
    """Sorted dedup of ``x`` (flattened).

    return_inverse=False -> y                (bare tensor, matches task/reference.py)
    return_inverse=True  -> (y, inverse)

    The whole computation runs in the device kernels; only the distinct count is
    handed to the host because the length of ``y`` is part of its metadata.
    """
    y, inverse = torch.ops.cann_bench.unique(x, return_inverse)
    if return_inverse:
        return y, inverse
    return y

def weight_quant_batch_matmul(x: torch.Tensor,
                              weight: torch.Tensor,
                              antiquantScale: torch.Tensor,
                              antiquantOffset=None,
                              bias=None) -> torch.Tensor:
    """y = x @ ((weight + antiquantOffset) * antiquantScale) + bias."""
    return torch.ops.cann_bench.weight_quant_batch_matmul(
        x, weight, antiquantScale, antiquantOffset, bias)

def gqa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        scaleValue: float = -1.0, is_causal: bool = False) -> torch.Tensor:
    """Grouped query attention over already-split heads."""
    return torch.ops.cann_bench.gqa(query, key, value, scaleValue, is_causal)

def grouped_matmul_swiglu_quant(x: torch.Tensor,
                                weight: torch.Tensor,
                                weight_scale: torch.Tensor,
                                x_scale: torch.Tensor,
                                group_list):
    """Fused grouped matmul + dequant + SwiGLU + per-token int8 requant.

    group_list is the cumsum (inclusive prefix sum) of the per-expert token counts, given as the
    operator host attribute (a plain sequence of ints).  Returns (y, y_scale).
    """
    return torch.ops.cann_bench.grouped_matmul_swiglu_quant(
        x, weight, weight_scale, x_scale, group_list)

def gru(x=None, weight_ih=None, weight_hh=None, *args, **kwargs):
    """Gated Recurrent Unit matching ``torch.nn.GRU`` (eval mode) semantics.

    Returns ``(y, hn)``.
    """
    slot = dict(_GRU_DEFAULTS)
    if x is None and "x" in kwargs:
        x = kwargs.pop("x")
    if weight_ih is None and "weight_ih" in kwargs:
        weight_ih = kwargs.pop("weight_ih")
    if weight_hh is None and "weight_hh" in kwargs:
        weight_hh = kwargs.pop("weight_hh")
    for key in list(kwargs.keys()):
        if key in slot:
            slot[key] = kwargs.pop(key)
    if kwargs:
        raise TypeError("gru() got unexpected keyword arguments: %s" % sorted(kwargs))

    pos = list(args)
    if pos:
        first = pos[0]
        is_int = isinstance(first, int) and not isinstance(first, bool)
        names = _GRU_EVAL_ORDER if is_int else _GRU_DOC_ORDER
        for name, value in zip(names, pos):
            slot[name] = value

    if x is None or weight_ih is None or weight_hh is None:
        raise TypeError("gru() requires x, weight_ih and weight_hh")

    weight_ih = list(weight_ih)
    weight_hh = list(weight_hh)
    bias_ih = _gru_tensor_list(slot["bias_ih"])
    bias_hh = _gru_tensor_list(slot["bias_hh"])
    h0 = _gru_tensor(slot["h0"])

    input_size = _gru_int(slot["inputSize"], 0)
    hidden_size = _gru_int(slot["hiddenSize"], 0)
    num_layers = _gru_int(slot["numLayers"], 1)
    if input_size <= 0:
        input_size = int(x.shape[-1])
    if hidden_size <= 0 and weight_hh:
        hidden_size = int(weight_hh[0].shape[1])
    if num_layers <= 0:
        num_layers = 1

    return torch.ops.cann_bench.gru(
        x,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        h0,
        int(input_size),
        int(hidden_size),
        int(num_layers),
        bool(slot["bias"]),
        bool(slot["batchFirst"]),
        float(slot["dropout"]),
        bool(slot["bidirectional"]),
    )

def lstm(x, weight_ih, weight_hh, *args, **kwargs):
    """LSTM operator entry point.

    Two calling conventions are accepted:

    * ``lstm(x, weight_ih, weight_hh, inputSize, hiddenSize, numLayers, bias,
      batchFirst, dropout, bidirectional, projSize, bias_ih, bias_hh, h0, c0)``
      (the ordering used by the registered torch prototype), and
    * ``lstm(x, weight_ih, weight_hh, bias_ih, bias_hh, h0, c0, inputSize=...,
      hiddenSize=..., ...)`` (tensor arguments first, attributes by keyword).
    """
    input_size = kwargs.pop("inputSize", None)
    hidden_size = kwargs.pop("hiddenSize", None)
    num_layers = kwargs.pop("numLayers", 1)
    use_bias = kwargs.pop("bias", True)
    batch_first = kwargs.pop("batchFirst", False)
    dropout = kwargs.pop("dropout", 0.0)
    bidirectional = kwargs.pop("bidirectional", False)
    proj_size = kwargs.pop("projSize", 0)
    bias_ih = kwargs.pop("bias_ih", None)
    bias_hh = kwargs.pop("bias_hh", None)
    h0 = kwargs.pop("h0", None)
    c0 = kwargs.pop("c0", None)

    vals = list(args)
    # ``inputSize`` is an int in the prototype ordering and a tensor list (or None)
    # in the tensor-first ordering, so it discriminates the two.
    proto_order = bool(vals) and isinstance(vals[0], int)
    if proto_order:
        if len(vals) > 0:
            input_size = int(vals[0])
        if len(vals) > 1:
            hidden_size = int(vals[1])
        if len(vals) > 2:
            num_layers = int(vals[2])
        if len(vals) > 3:
            use_bias = bool(vals[3])
        if len(vals) > 4:
            batch_first = bool(vals[4])
        if len(vals) > 5:
            dropout = float(vals[5])
        if len(vals) > 6:
            bidirectional = bool(vals[6])
        if len(vals) > 7:
            proj_size = int(vals[7])
        if len(vals) > 8:
            bias_ih = vals[8]
        if len(vals) > 9:
            bias_hh = vals[9]
        if len(vals) > 10:
            h0 = vals[10]
        if len(vals) > 11:
            c0 = vals[11]
    else:
        if len(vals) > 0:
            bias_ih = vals[0]
        if len(vals) > 1:
            bias_hh = vals[1]
        if len(vals) > 2:
            h0 = vals[2]
        if len(vals) > 3:
            c0 = vals[3]

    if hidden_size is None:
        hidden_size = weight_hh[0].size(1) if weight_hh else 0
    if input_size is None:
        input_size = x.size(2)

    return torch.ops.cann_bench.lstm(
        x,
        _as_tensor_list(weight_ih),
        _as_tensor_list(weight_hh),
        int(input_size),
        int(hidden_size),
        int(num_layers),
        bool(use_bias),
        bool(batch_first),
        float(dropout),
        bool(bidirectional),
        int(proj_size),
        _as_tensor_list(bias_ih),
        _as_tensor_list(bias_hh),
        h0,
        c0,
    )

def mha(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        scaleValue: float = -1.0, is_causal: bool = False) -> torch.Tensor:
    """y = softmax(Q @ K^T * scaleValue) @ V for pre-split-head Q/K/V [B, S, N, D]."""
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
    """MLA attention: softmax(concat(q_nope, q_rope) @ concat(k_nope, k_rope)^T * scale) @ v."""
    return torch.ops.cann_bench.mla(
        q_nope, q_rope, k_nope, k_rope, v, numKVHeads, scaleValue, inputLayout, is_causal
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
    """MLA prolog: fused Q/K projections + RMSNorm + RoPE.

    Returns (query, query_rope, c_kv, k_rope).
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

def sparse_flash_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                           sparseIndices: torch.Tensor, scaleValue: float,
                           inputLayout: str = "BSND", is_causal: bool = False) -> torch.Tensor:
    """Sparse attention: y = softmax(Q @ gather(K, idx)^T * scale) @ gather(V, idx)."""
    return torch.ops.cann_bench.sparse_flash_attention(
        query, key, value, sparseIndices, scaleValue, inputLayout, is_causal)

def gcd(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """y = gcd(x1, x2), elementwise with numpy broadcasting; y.dtype == x1.dtype."""
    return torch.ops.cann_bench.gcd(x1, x2)

def resize_bilinear(
    x: torch.Tensor,
    output_size=None,
    align_corners: bool = False,
    scale_factor=None,
) -> torch.Tensor:
    """Bilinear resize of a 4D (N, C, H, W) tensor; output dtype == x.dtype."""
    if output_size is not None:
        output_size = [int(v) for v in output_size]
    if scale_factor is not None:
        scale_factor = [float(v) for v in scale_factor]
    return torch.ops.cann_bench.resize_bilinear(x, output_size, align_corners, scale_factor)

def arg_max(input: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """Return the int64 indices of the maximum along ``dim`` (first occurrence on ties)."""
    return torch.ops.cann_bench.arg_max(input, dim, keepdim)

def cummin(input: torch.Tensor, dim: int):
    """Cumulative minimum along ``dim``.

    Returns a tuple ``(values, indices)`` where ``values`` has the same shape and
    dtype as ``input`` and ``indices`` is an int64 tensor holding the argmin along
    the reduction axis (torch.cummin semantics).
    """
    return torch.ops.cann_bench.cummin(input, dim)

def scatter(data: torch.Tensor, dim: int, indices: torch.Tensor, updates: torch.Tensor,
            reduce: str = None) -> torch.Tensor:
    """Scatter ``updates`` into ``data`` along ``dim``.

    ``reduce`` is one of ``None``/``"update"``, ``"add"``, ``"multiply"``, ``"amin"``, ``"amax"``.
    """
    return torch.ops.cann_bench.scatter(data, dim, indices, updates, reduce)

def grid_sampler_3d(
    x: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """y = grid_sample(x, grid); x is (N, C, D, H, W), grid is (N, Do, Ho, Wo, 3)."""
    return torch.ops.cann_bench.grid_sampler_3d(
        x, grid, interpolation_mode, padding_mode, align_corners
    )

def apply_adam_w(
    var: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    epsilon: float = 1e-8,
    step: int = 1,
    maximize: bool = False,
) -> torch.Tensor:
    """Applied AdamW (decoupled weight decay); output dtype == var.dtype."""
    return torch.ops.cann_bench.apply_adam_w(
        var, grad, m, v, lr, beta1, beta2, weight_decay, epsilon, step, maximize
    )

def apply_rotary_pos_emb(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor,
                         sin: torch.Tensor, layout: int = 0,
                         rotaryMode: str = "half"):
    """Rotary position embedding applied to query and key.

    layout 0: query/key are (B, S, N, D); layout 1: query/key are (B, N, S, D).
    rotaryMode "half" (contiguous halves) or "interleaved" (even/odd pairs).
    """
    return torch.ops.cann_bench.apply_rotary_pos_emb(query, key, cos, sin, layout, rotaryMode)

def cross_entropy_loss(input: torch.Tensor, target: torch.Tensor,
                       reduction: str = "mean", ignore_index: int = -100) -> torch.Tensor:
    """Cross entropy loss for hard labels; matches torch.nn.functional.cross_entropy."""
    return torch.ops.cann_bench.cross_entropy_loss(input, target, reduction, ignore_index)

def gather(x: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """y = torch.gather(x, dim, index); shape == index.shape, dtype == x.dtype."""
    return torch.ops.cann_bench.gather(x, index, dim)

def rms_norm(x: torch.Tensor = None, gamma: torch.Tensor = None,
             epsilon: float = 1e-6, **kwargs) -> torch.Tensor:
    """RMS normalization: y = x / sqrt(mean(x^2) + eps) * gamma.

    The tensor arguments are also accepted under the dispatcher's alternative
    spellings (``input`` / ``weight``) because the benchmark entry point may
    pass them either way.
    """
    if x is None:
        x = kwargs.pop("input", None)
    if gamma is None:
        gamma = kwargs.pop("weight", None)
    if epsilon == 1e-6:
        eps = kwargs.pop("eps", None)
        if eps is not None:
            epsilon = eps
    if x is None or gamma is None:
        raise TypeError("rms_norm() missing required argument: 'x' or 'gamma'")
    return torch.ops.cann_bench.rms_norm(x, gamma, float(epsilon))

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """y = exp(x - max) / sum(exp(x - max)) along `dim`."""
    return torch.ops.cann_bench.softmax(x, dim)

def maximum(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.ops.cann_bench.maximum(x1, x2)
