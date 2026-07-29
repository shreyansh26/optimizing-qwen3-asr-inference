"""Out-of-tree vLLM optimization plugin for Qwen3-ASR."""

from __future__ import annotations

import difflib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from audio_cpu_maxseqlen_patch import install_audio_cpu_maxseqlen_patch
from audio_encoder_cudagraph_patch import (
    install_requested_audio_cudagraph_patches,
)
from qk_mrope_kv_cache_fusion_patch import install_qk_mrope_kv_cache_fusion_patch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerTensorOnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
    kFp8StaticTensorSym,
)


logger = init_logger("vllm.static_fp8_json")

if os.environ.get("ASR_QK_MROPE_KV_CACHE_FUSION", "0") == "1":
    install_qk_mrope_kv_cache_fusion_patch()
if os.environ.get("ASR_AUDIO_CPU_MAXSEQLEN", "0") == "1":
    install_audio_cpu_maxseqlen_patch()
install_requested_audio_cudagraph_patches()

SCALES_ENV = "ASR_FP8_STATIC_SCALES_JSON"
EXPECTED_FORMAT = "qwen3_asr_fp8_static_activation_scales"
_MATCHED_PREFIXES: set[str] = set()
_MATERIALIZED_PREFIXES: set[str] = set()
_ALL_PREFIXES: set[str] = set()


def _write_coverage(raw_path: str | None) -> None:
    if not raw_path or not _ALL_PREFIXES:
        return

    path = Path(raw_path.format(pid=os.getpid())).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scale_count": len(_ALL_PREFIXES),
        "matched_count": len(_MATCHED_PREFIXES),
        "materialized_count": len(_MATERIALIZED_PREFIXES),
        "matched_prefixes": sorted(_MATCHED_PREFIXES),
        "materialized_prefixes": sorted(_MATERIALIZED_PREFIXES),
        "unused_prefixes": sorted(_ALL_PREFIXES - _MATCHED_PREFIXES),
        "pid": os.getpid(),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _log_progress(label: str, prefixes: set[str], total: int) -> None:
    count = len(prefixes)
    if count == total:
        logger.info("%s all %d static FP8 activation scales", label, total)
    elif count == 1 or count % 25 == 0:
        logger.info("%s %d/%d static FP8 activation scales", label, count, total)


def _to_vllm_prefix(hf_prefix: str) -> str:
    """Map the recorder's Transformers module names to Qwen3-ASR vLLM names."""
    if hf_prefix.startswith("thinker.model."):
        return "language_model.model." + hf_prefix.removeprefix("thinker.model.")

    if hf_prefix.startswith("thinker.audio_tower."):
        vllm_prefix = hf_prefix.removeprefix("thinker.")
        # The HF model exposes q_proj/k_proj/v_proj. The recorder calls their
        # fused scale qkv_proj, while vLLM names this packed layer qkv.
        return vllm_prefix.replace(".self_attn.qkv_proj", ".self_attn.qkv")

    if hf_prefix.startswith("thinker.lm_head."):
        return "language_model.lm_head." + hf_prefix.removeprefix(
            "thinker.lm_head."
        )

    raise ValueError(f"Unsupported calibration module prefix: {hf_prefix}")


def _load_scale_table(path: Path) -> tuple[dict[str, float], str]:
    with path.open() as handle:
        artifact = json.load(handle)

    if artifact.get("format") != EXPECTED_FORMAT:
        raise ValueError(
            f"Unexpected scale JSON format {artifact.get('format')!r}; "
            f"expected {EXPECTED_FORMAT!r}"
        )
    if artifact.get("version") != 1:
        raise ValueError(
            f"Unsupported scale JSON version {artifact.get('version')!r}; expected 1"
        )

    fused_modules = artifact.get("vllm_fused_modules")
    if not isinstance(fused_modules, dict) or not fused_modules:
        raise ValueError("Scale JSON has no non-empty 'vllm_fused_modules' mapping")

    scale_table: dict[str, float] = {}
    for hf_prefix, stats in fused_modules.items():
        if not isinstance(stats, dict) or "scale" not in stats:
            raise ValueError(f"Missing scale for calibration module {hf_prefix!r}")
        scale = float(stats["scale"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid scale {scale!r} for {hf_prefix!r}")

        vllm_prefix = _to_vllm_prefix(hf_prefix)
        if vllm_prefix in scale_table:
            raise ValueError(f"Duplicate mapped vLLM module prefix: {vllm_prefix}")
        scale_table[vllm_prefix] = scale

    return scale_table, str(artifact.get("model", "unknown"))


class StaticPerTensorOnlineLinearMethod(Fp8PerTensorOnlineLinearMethod):
    """Online FP8 weights with one fixed activation scale per linear layer."""

    def __init__(
        self,
        module_prefix: str,
        input_scale: float,
        expected_scale_count: int,
        coverage_path: str | None,
    ):
        super().__init__()
        self.module_prefix = module_prefix
        self.static_input_scale = input_scale
        self.expected_scale_count = expected_scale_count
        self.coverage_path = coverage_path
        self.activation_quant_key = kFp8StaticTensorSym

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        super().process_weights_after_loading(layer)
        layer.input_scale = torch.tensor(
            self.static_input_scale,
            dtype=torch.float32,
            device=layer.weight.device,
        )
        _MATERIALIZED_PREFIXES.add(self.module_prefix)
        _write_coverage(self.coverage_path)
        _log_progress(
            "Materialized", _MATERIALIZED_PREFIXES, self.expected_scale_count
        )


@register_quantization_config("fp8_static_json")
class StaticJsonFp8Config(Fp8Config):
    """Load BF16 weights as FP8 and use calibrated static activation scales."""

    def __init__(self) -> None:
        super().__init__(
            is_checkpoint_fp8_serialized=False,
            activation_scheme="static",
        )

        raw_path = os.environ.get(SCALES_ENV)
        if not raw_path:
            raise ValueError(f"{SCALES_ENV} must point to a calibration JSON file")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Static FP8 scale JSON not found: {path}")

        all_scales, calibrated_model = _load_scale_table(path)
        self.scale_table = {
            prefix: scale
            for prefix, scale in all_scales.items()
            if prefix.startswith("language_model.model.")
        }
        ignored_scale_count = len(all_scales) - len(self.scale_table)
        if not self.scale_table:
            raise ValueError("Scale JSON has no language-model linear layers")
        self.coverage_path = os.environ.get("ASR_FP8_STATIC_COVERAGE_JSON")
        _ALL_PREFIXES.update(self.scale_table)
        logger.info(
            "Loaded %d calibrated static FP8 decoder scales for %s from %s; "
            "ignored %d audio-tower scales because vLLM leaves that tower unquantized",
            len(self.scale_table),
            calibrated_model,
            path,
            ignored_scale_count,
        )

    @classmethod
    def get_name(cls) -> str:
        return "fp8_static_json"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "StaticJsonFp8Config":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Any | None:
        _ALL_PREFIXES.update(self.scale_table)

        if not isinstance(layer, LinearBase):
            return super().get_quant_method(layer, prefix)

        if is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            return UnquantizedLinearMethod()

        input_scale = self.scale_table.get(prefix)
        if input_scale is None:
            nearby = difflib.get_close_matches(prefix, self.scale_table, n=3)
            hint = f" Closest calibrated names: {nearby}" if nearby else ""
            raise KeyError(
                f"No calibrated static FP8 activation scale for vLLM layer "
                f"{prefix!r}.{hint}"
            )

        total = len(self.scale_table)
        method = StaticPerTensorOnlineLinearMethod(
            prefix, input_scale, total, self.coverage_path
        )
        method.marlin_input_dtype = get_marlin_input_dtype(prefix)

        _MATCHED_PREFIXES.add(prefix)
        _write_coverage(self.coverage_path)
        _log_progress("Matched", _MATCHED_PREFIXES, total)

        return method
