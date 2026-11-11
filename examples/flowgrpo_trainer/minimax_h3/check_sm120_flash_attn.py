# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Preflight for the sm120 FLASH_ATTN rollout: run it on every rollout node.

Checks that CuTe FlashAttention-4 imports on this GPU, that vLLM-Omni's
``FLASH_ATTN`` backend resolves to it, and that FA4 matches PyTorch SDPA on
MiniMax H3 attention shapes (head_dim 128; 28 heads per rank at rollout TP=2,
56 heads for the unsharded Actor). Exits non-zero on any failure.

Usage::

    python3 examples/flowgrpo_trainer/minimax_h3/check_sm120_flash_attn.py
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn.functional as F


def _bench(fn, iters: int = 10) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[4096, 12288])
    parser.add_argument("--heads", type=int, nargs="+", default=[28])
    parser.add_argument("--fwd-tol", type=float, default=3e-2)
    parser.add_argument("--grad-tol", type=float, default=1e-1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available")
        return 1
    capability = torch.cuda.get_device_capability()
    print(f"device={torch.cuda.get_device_name()} capability={capability} torch={torch.__version__}")
    if capability[0] < 10:
        print("FAIL: FA4 rollout targets Blackwell (compute capability >= 10)")
        return 1

    try:
        from flash_attn.cute import flash_attn_varlen_func
    except Exception as exc:  # noqa: BLE001 - report any import/ABI failure
        print(f"FAIL: cannot import flash_attn.cute (pip install flash-attn-4): {exc!r}")
        return 1

    try:
        import vllm_omni.diffusion.attention.backends.utils.fa as vllm_omni_fa

        selected = getattr(vllm_omni_fa.flash_attn_varlen_func, "__module__", None)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: cannot import vLLM-Omni FA utilities: {exc!r}")
        return 1
    print(f"vLLM-Omni FLASH_ATTN varlen implementation: {selected}")
    if not selected or not selected.startswith("flash_attn.cute"):
        print("FAIL: vLLM-Omni FLASH_ATTN does not resolve to flash_attn.cute on this GPU")
        return 1

    torch.manual_seed(0)
    head_dim = 128
    for num_heads in args.heads:
        for seq_len in args.seq_lens:
            q, k, v = (
                torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                for _ in range(3)
            )
            cu_seqlens = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)

            def fa4(q=q, k=k, v=v, cu_seqlens=cu_seqlens, seq_len=seq_len):
                out = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=seq_len,
                    max_seqlen_k=seq_len,
                )
                return out[0] if isinstance(out, tuple) else out

            def sdpa(q=q, k=k, v=v):
                layout = [x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)]
                return F.scaled_dot_product_attention(*layout)[0].transpose(0, 1)

            out_fa4, out_sdpa = fa4(), sdpa()
            fwd_err = (out_fa4.float() - out_sdpa.float()).abs().max().item()
            grad_out = torch.randn_like(out_fa4)
            out_fa4.backward(grad_out)
            dq_fa4 = q.grad.clone()
            q.grad = None
            out_sdpa.backward(grad_out)
            dq_err = (dq_fa4.float() - q.grad.float()).abs().max().item()
            q.grad = k.grad = v.grad = None

            with torch.no_grad():
                ms_fa4, ms_sdpa = _bench(fa4), _bench(sdpa)
            status = "ok" if fwd_err < args.fwd_tol and dq_err < args.grad_tol else "MISMATCH"
            print(
                f"heads={num_heads} seq={seq_len}: max|out diff|={fwd_err:.3e} max|dq diff|={dq_err:.3e} "
                f"fwd fa4={ms_fa4:.2f}ms sdpa(no mask)={ms_sdpa:.2f}ms [{status}]"
            )
            if status != "ok":
                print("FAIL: FA4 does not match SDPA")
                return 1

    print("SM120_FLASH_ATTN_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
