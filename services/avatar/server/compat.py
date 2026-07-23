"""Windows / single-GPU compatibility shim for vendored SoulX-FlashHead.

xfuser is only needed for multi-GPU sequence parallelism and does not install
on Windows; flash_head_model.py imports it at module level, so we stub it
before any flash_head import.
"""

from __future__ import annotations

import sys
import types


def _install_xfuser_stub() -> None:
    try:
        import xfuser  # noqa: F401
        return
    except ImportError:
        pass

    def _unavailable(*_a, **_k):
        raise RuntimeError("xfuser stub: multi-GPU parallelism not available on this host")

    distributed = types.ModuleType("xfuser.core.distributed")
    distributed.get_sequence_parallel_rank = _unavailable
    distributed.get_sequence_parallel_world_size = _unavailable
    distributed.get_sp_group = _unavailable
    distributed.init_distributed_environment = _unavailable
    distributed.initialize_model_parallel = _unavailable
    distributed.get_world_group = _unavailable

    long_ctx = types.ModuleType("xfuser.core.long_ctx_attention")

    class xFuserLongContextAttention:  # noqa: N801 — matches upstream name
        def __init__(self, *_a, **_k):
            _unavailable()

    long_ctx.xFuserLongContextAttention = xFuserLongContextAttention

    core = types.ModuleType("xfuser.core")
    core.distributed = distributed
    core.long_ctx_attention = long_ctx

    root = types.ModuleType("xfuser")
    root.core = core

    sys.modules["xfuser"] = root
    sys.modules["xfuser.core"] = core
    sys.modules["xfuser.core.distributed"] = distributed
    sys.modules["xfuser.core.long_ctx_attention"] = long_ctx
