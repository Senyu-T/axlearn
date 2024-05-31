# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# jax-ml/jax-triton:
# Copyright 2023 The jax_triton Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests GPU FlashAttention kernels.

Currently tested on A100/H100.
"""
# pylint: disable=wrong-import-position
import functools
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import chex
import jax
import jax.numpy as jnp
import pytest
from jax.experimental.pallas.ops.attention import mha as pallas_mha

from axlearn.common.flash_attention.gpu_attention import flash_attention
from axlearn.common.flash_attention.utils import flash_attention_implementation, mha_reference


@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,per_head_dim",
    [
        (1, 384, 1, 64),
        (2, 384, 2, 64),
        (1, 384, 1, 64),
        (2, 384, 2, 64),
        (1, 384, 8, 64),
        (2, 384, 8, 64),
    ],
)
@pytest.mark.parametrize("block_size", [128, 64])
@pytest.mark.parametrize("use_fwd", [True, False])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("sm_scale", [1.0, 0.123])
@pytest.mark.parametrize("bias_type", ["none", "matrix", "vector"])
@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Test only runs on GPU.")
def test_fwd_against_ref(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    per_head_dim: int,
    block_size: int,
    use_fwd: bool,
    causal: bool,
    sm_scale: float,
    bias_type: str,
):
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16)
    k = jax.random.normal(k2, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16)
    v = jax.random.normal(k3, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16)

    if bias_type == "matrix":
        bias = jax.random.normal(k4, (batch_size, num_heads, seq_len, seq_len), dtype=jnp.float16)
    elif bias_type == "vector":
        segment_left = jnp.ones((batch_size, seq_len // 2), dtype=jnp.int32)
        segment_right = jnp.zeros((batch_size, seq_len // 2), dtype=jnp.int32)
        bias = jnp.concatenate([segment_left, segment_right], axis=-1)
    else:
        bias = None
    # Make sure that it is running on GPU.
    assert str(q.devices()) == "{cuda(id=0)}"

    if use_fwd:

        @jax.jit
        def impl(q, k, v, bias):
            fn = functools.partial(
                flash_attention,
                block_q=block_size,
                block_k=block_size,
                causal=causal,
                softmax_scale=sm_scale,
            )
            out, _ = jax.vjp(fn, q, k, v, bias)
            return out

    else:
        impl = functools.partial(
            flash_attention,
            block_q=block_size,
            block_k=block_size,
            causal=causal,
            softmax_scale=sm_scale,
        )

    o = impl(q, k, v, bias)
    o_ref = mha_reference(q, k, v, bias, causal=causal, softmax_scale=sm_scale)
    chex.assert_trees_all_close(o, o_ref, atol=0.05)


@pytest.mark.parametrize(
    "batch_size,num_heads,seq_len,per_head_dim",
    [
        (1, 1, 384, 64),
        (2, 2, 384, 64),
        (1, 1, 384, 64),
        (2, 2, 384, 64),
        (1, 8, 384, 64),
        (2, 8, 384, 64),
    ],
)
@pytest.mark.parametrize("bias_type", ["none", "matrix", "vector"])
@pytest.mark.parametrize("block_size", [128, 64])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Test only runs on GPU.")
def test_bwd_against_ref(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    per_head_dim: int,
    bias_type: str,
    block_size: int,
    causal: bool,
):
    q = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
    )
    k = jax.random.normal(
        jax.random.PRNGKey(1), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
    )
    v = jax.random.normal(
        jax.random.PRNGKey(2), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
    )

    if bias_type == "matrix":
        bias = jax.random.normal(
            jax.random.PRNGKey(3), (batch_size, num_heads, seq_len, seq_len), dtype=jnp.float16
        )
    elif bias_type == "vector":
        segment_left = jnp.ones((batch_size, seq_len // 2), dtype=jnp.int32)
        segment_right = jnp.zeros((batch_size, seq_len // 2), dtype=jnp.int32)
        bias = jnp.concatenate([segment_left, segment_right], axis=-1)
    else:
        bias = None
    # Make sure that it is running on GPU.
    assert str(q.devices()) == "{cuda(id=0)}"

    sm_scale = q.shape[-1] ** -0.5

    # Compare outputs.
    jax_out = flash_attention(q, k, v, bias, causal=causal, softmax_scale=sm_scale)
    jax_ref_out = mha_reference(q, k, v, bias, causal=causal, softmax_scale=sm_scale)
    chex.assert_trees_all_close(jax_out, jax_ref_out, atol=0.005)

    def fn(q, k, v, bias):
        return flash_attention(
            q,
            k,
            v,
            bias,
            causal=causal,
            softmax_scale=sm_scale,
            block_q=block_size,
            block_k=block_size,
        ).sum()

    def ref_fn(q, k, v, bias):
        return mha_reference(q, k, v, bias, causal=causal, softmax_scale=sm_scale).sum()

    # Compare gradients.
    jax_grads = jax.grad(fn, argnums=(0, 1, 2))(q, k, v, bias)
    jax_ref_grads = jax.grad(ref_fn, argnums=(0, 1, 2))(q, k, v, bias)
    chex.assert_trees_all_close(jax_grads, jax_ref_grads, atol=0.05)


# We also include a test for Triton with Pallas, to cross validate the triton
# compatibility with our own implementation.
@pytest.mark.parametrize(
    "batch_size,num_heads,seq_len,per_head_dim",
    [
        (2, 2, 384, 64),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("bias_type", ["none", "vector"])
@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Test only runs on GPU.")
def test_mha_against_pallas_ref(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    per_head_dim: int,
    causal: bool,
    bias_type: str,
):
    q = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
    )
    k = jax.random.normal(
        jax.random.PRNGKey(1), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
    )
    v = jax.random.normal(
        jax.random.PRNGKey(2), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
    )
    # Make sure that it is running on GPU.
    assert str(q.devices()) == "{cuda(id=0)}"

    sm_scale = q.shape[-1] ** -0.5
    if bias_type == "vector":
        segment_left = jnp.ones((batch_size, seq_len // 2), dtype=jnp.int32)
        segment_right = jnp.zeros((batch_size, seq_len // 2), dtype=jnp.int32)
        segment_ids = jnp.concatenate([segment_left, segment_right], axis=-1)
    else:
        segment_ids = None
    # Compare outputs.
    jax_out = mha_reference(q, k, v, bias=segment_ids, causal=causal, softmax_scale=sm_scale)
    jax_ref_out = pallas_mha(q, k, v, segment_ids=segment_ids, causal=causal, sm_scale=sm_scale)
    chex.assert_trees_all_close(jax_out, jax_ref_out, atol=0.005, rtol=1e-5)

    def fn(q, k, v):
        return mha_reference(q, k, v, bias=segment_ids, causal=causal, softmax_scale=sm_scale).sum()

    def ref_fn(q, k, v):
        return pallas_mha(q, k, v, segment_ids=segment_ids, causal=causal, sm_scale=sm_scale).sum()

    # Compare gradients.
    jax_grads = jax.grad(fn, argnums=(0, 1, 2))(q, k, v)
    jax_ref_grads = jax.grad(ref_fn, argnums=(0, 1, 2))(q, k, v)
    chex.assert_trees_all_close(jax_grads, jax_ref_grads, atol=0.05, rtol=1e-5)

# Causal = True, Vector
@pytest.mark.parametrize(
    "batch_size,num_heads,seq_len,per_head_dim",
    [
        (2, 8, 384, 64),
        (4, 12, 2048, 64),
    ],
)
@pytest.mark.parametrize("bias_type", ["none", "1HSS"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype, atol", [(jnp.float16,0.005), (jnp.bfloat16, 0.01)])
@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Test only runs on GPU.")
def test_mha_against_te_ref(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    per_head_dim: int,
    causal: bool,
    bias_type: str,
    dtype: jnp.dtype,
    atol: float,
):
    # pytype: disable=import-error
    # pylint: disable=import-outside-toplevel
    # Checking if test inputs are Valid.
    from transformer_engine.jax.cpp_extensions import FusedAttnHelper
    from transformer_engine.jax.fused_attn import AttnBiasType, AttnMaskType, QKVLayout
    from transformer_engine_jax import NVTE_Fused_Attn_Backend

    # PRE_SCALE_BIAS is not yet supported in any valid background.
    attn_bias_type = AttnBiasType.NO_BIAS if bias_type == "none" else AttnBiasType.POST_SCALE_BIAS
    # We only test on causal mask or no mask.
    attn_mask_type = AttnMaskType.NO_MASK if causal == "false" else AttnMaskType.CAUSAL_MASK
    # Currently we only test on BSHD_BSHD_BSHD layout.
    nvte_backend = FusedAttnHelper(dtype, dtype, QKVLayout.BSHD_BSHD_BSHD.value,
                                   attn_bias_type.value, attn_mask_type.value, 0.0,
                                   num_heads, num_heads, seq_len, seq_len,
                                   per_head_dim).get_fused_attn_backend()
    if nvte_backend == NVTE_Fused_Attn_Backend.NVTE_No_Backend:
        pytest.skip("Unsupported inputs combination or device compute capability")
    if attn_bias_type == AttnBiasType.POST_SCALE_BIAS and \
       nvte_backend == NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen:
        pytest.skip("B1SS, BHSS, 11SS bias are only supported for F16_arbitrary_seqlen backend")


    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(k2, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.bfloat16)
    v = jax.random.normal(k3, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.bfloat16)

    # TE only supports PostScaleBias in shape (1, H, S, S) for grad compute
    bias = None if bias_type == "none"  \
                else jax.random.normal(k4, (1, num_heads, seq_len, seq_len), dtype=dtype)

    sm_scale = q.shape[-1] ** -0.5

    jax_out = mha_reference(q, k, v, bias=bias, causal=causal, softmax_scale=sm_scale)
    assert "NVTE_FUSED_ATTN" in os.environ,  \
        "Must Enable NVTE_FUSED_ATTN to Test TransformerEngine Output"
    te_attn = flash_attention_implementation(
        backend="gpu", causal=causal, softmax_scale=sm_scale,
    )
    te_out = te_attn(q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16), None)
    chex.assert_trees_all_close(jax_out, te_out, atol=atol)


    # Backward
    if attn_bias_type != AttnBiasType.NO_BIAS:
        pytest.skip("Only test backward pass without bias")

    def fn(q, k, v, bias):
        return mha_reference(q, k, v, bias, causal=causal, softmax_scale=sm_scale).sum()

    def ref_fn(q, k, v, bias):
        return te_attn(q, k, v, bias).sum()

    jax_grads = jax.grad(fn, argnums=(0, 1, 2, 3))(q, k, v, bias)
    te_grads = jax.grad(ref_fn, argnums=(0, 1, 2, 3))(q, k, v, bias)
    chex.assert_trees_all_close(jax_grads, te_grads, atol=0.05, rtol=1e-5)
