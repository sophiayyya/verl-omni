# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""FA2/FA3/FA4 availability checks and validation for matched actor/rollout attention."""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

logger = logging.getLogger(__name__)

ACTOR_FA2_BACKEND = "flash_varlen_hub"
ACTOR_FA3_BACKEND = "_flash_3_varlen_hub"
ACTOR_NATIVE_BACKEND = "native"
ROLLOUT_SDPA_BACKEND = "TORCH_SDPA"

# Keep in sync with vllm-omni diffusion attention backends for FA train/rollout pairs.
FA_ROLLOUT_BACKENDS = ("FLASH_ATTN", "FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB")
ACTOR_BACKENDS = (ACTOR_FA2_BACKEND, ACTOR_FA3_BACKEND, ACTOR_NATIVE_BACKEND, "_native_npu")
ROLLOUT_BACKENDS = FA_ROLLOUT_BACKENDS + (ROLLOUT_SDPA_BACKEND,)


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        # find_spec imports parent packages for dotted names; a broken parent
        # (e.g. an ABI-mismatched flash_attn wheel) means the module is unusable.
        return False


def actor_fa_available() -> bool:
    return _module_available("kernels")


def _cuda_capability() -> tuple[int, int] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability()
        return int(major), int(minor)
    except Exception:
        return None


def _cuda_supports_rollout_fa() -> bool:
    """Ampere/Hopper: the FA2/FA3 wheels that back ``FLASH_ATTN`` ship kernels for these."""
    capability = _cuda_capability()
    if capability is None:
        return False
    compute_capability = capability[0] + capability[1] / 10.0
    return 8.0 <= compute_capability < 10.0


def rollout_fa4_available() -> bool:
    """True on Blackwell (sm100/sm110/sm120) when CuTe FlashAttention-4 is importable.

    On compute capability >= 10, vLLM-Omni's ``FLASH_ATTN`` backend prefers
    ``flash_attn.cute`` (FA4) over Hopper-only FA3 wheels, which would otherwise
    import fine and then fail at launch with "no kernel image is available".
    FA4 ships forward and backward kernels for sm120 (RTX PRO Blackwell class).
    """
    capability = _cuda_capability()
    if capability is None or capability[0] < 10:
        return False
    return _module_available("flash_attn.cute")


def rollout_fa_available() -> bool:
    """True when local FA packages can back ``FLASH_ATTN`` rollout."""
    if _cuda_supports_rollout_fa():
        return any(_module_available(name) for name in ("fa3_fwd_interface", "flash_attn"))
    return rollout_fa4_available()


def fa_available() -> bool:
    """True when the default FA pair (FA3 kernels-hub Actor + ``FLASH_ATTN`` rollout) runs locally.

    The Actor FA backends load kernels-hub FA2/FA3 builds, and the FA3 build has
    no Blackwell kernels, so a Blackwell GPU with FA4 rollout alone does not
    qualify; see :func:`rollout_fa4_available` for that case.
    """
    return actor_fa_available() and _cuda_supports_rollout_fa() and rollout_fa_available()


def validate_attention_consistency(config: Any) -> None:
    """Validate that rollout and training attention backends match.

    Rules:
        - If the training engine is VeOmni, skip validation.
        - If ``attn_backend`` is ``flash_varlen_hub`` or ``_flash_3_varlen_hub``
          (FA2/FA3), rollout must be one of ``FA_ROLLOUT_BACKENDS`` (default
          ``FLASH_ATTN_3_HUB`` for kernels FA3 train/rollout consistency).
        - If ``attn_backend`` is ``native`` or ``_native_npu``, rollout must be
          ``TORCH_SDPA``.

    Raises:
        ValueError: If the rollout attention backend does not match the training
            attention backend.
    """
    actor_cfg = config.actor_rollout_ref.actor
    strategy = actor_cfg.get("strategy") if hasattr(actor_cfg, "get") else None
    if strategy == "veomni":
        logger.warning("strategy=veomni: attention consistency is not validated; ensure backends match manually.")
        return  # VeOmni engine manages its own attention independently

    model_cfg = config.actor_rollout_ref.model
    attn_backend = model_cfg.get("attn_backend", ACTOR_FA3_BACKEND)
    rollout_backend = config.actor_rollout_ref.rollout.get("rollout_attn_backend")

    if attn_backend not in ACTOR_BACKENDS:
        raise ValueError(f"Unknown attn_backend={attn_backend!r}. Available options: {list(ACTOR_BACKENDS)}.")
    if rollout_backend not in ROLLOUT_BACKENDS:
        raise ValueError(
            f"Unknown rollout_attn_backend={rollout_backend!r}. Available options: {list(ROLLOUT_BACKENDS)}."
        )

    if attn_backend in (ACTOR_FA2_BACKEND, ACTOR_FA3_BACKEND):
        if rollout_backend in FA_ROLLOUT_BACKENDS:
            return
        expected = ", ".join(FA_ROLLOUT_BACKENDS)
    elif attn_backend in (ACTOR_NATIVE_BACKEND, "_native_npu"):
        expected = ROLLOUT_SDPA_BACKEND
    if rollout_backend != expected:
        raise ValueError(
            f"Attention backend mismatch: attn_backend={attn_backend!r} requires "
            f"rollout_attn_backend={expected!r}, but got {rollout_backend!r}. "
            "Both must use the same attention implementation. "
            "Set rollout_attn_backend via --diffusion-attention-backend flag "
            "or in the rollout config."
        )
