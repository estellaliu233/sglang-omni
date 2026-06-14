# SPDX-License-Identifier: Apache-2.0
"""CLI override + config-default tests for Higgs TTS decode mode.

Higgs TTS defaults to async decode for throughput, while ``--decode-mode
async|sync`` can force or disable that mode explicitly. Omitting
``--decode-mode`` preserves the pipeline default.
"""

from __future__ import annotations

import pytest
import typer

from sglang_omni.cli.serve import (
    apply_decode_mode_cli_overrides,
    apply_generation_server_args_cli_override,
)
from sglang_omni.config import PipelineConfig, StageConfig, resolve_stage_factory_args
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig


def _stage_args(config: PipelineConfig, stage_name: str) -> dict[str, object]:
    stage = next(s for s in config.stages if s.name == stage_name)
    return resolve_stage_factory_args(stage, config)


def _tts_engine_args(config: PipelineConfig) -> dict[str, object]:
    return _stage_args(config, "tts_engine")


def test_decode_mode_default_config_is_async():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    assert _tts_engine_args(config)["enable_async_decode"] is True


def test_decode_mode_cli_can_force_sync_and_async():
    config = HiggsTtsPipelineConfig(model_path="dummy")

    apply_decode_mode_cli_overrides(
        config, decode_mode="sync", async_lookahead_min_batch_size=None
    )
    assert _tts_engine_args(config)["enable_async_decode"] is False

    apply_decode_mode_cli_overrides(
        config, decode_mode="async", async_lookahead_min_batch_size=None
    )
    assert _tts_engine_args(config)["enable_async_decode"] is True


def test_decode_mode_absent_preserves_config_default():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    apply_decode_mode_cli_overrides(
        config, decode_mode=None, async_lookahead_min_batch_size=None
    )
    assert _tts_engine_args(config)["enable_async_decode"] is True


def test_async_lookahead_min_batch_size_override_applies_without_mode_toggle():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    apply_decode_mode_cli_overrides(
        config, decode_mode=None, async_lookahead_min_batch_size=4
    )
    args = _tts_engine_args(config)
    assert args["enable_async_decode"] is True
    assert args["async_decode_min_batch_size"] == 4


def test_async_lookahead_min_batch_size_must_be_positive():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter):
        apply_decode_mode_cli_overrides(
            config, decode_mode="async", async_lookahead_min_batch_size=0
        )


def test_async_lookahead_min_batch_size_rejected_with_sync_mode():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter, match="cannot be combined"):
        apply_decode_mode_cli_overrides(
            config, decode_mode="sync", async_lookahead_min_batch_size=4
        )


def test_decode_mode_cli_invalid_mode_rejected():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter):
        apply_decode_mode_cli_overrides(
            config, decode_mode="bogus", async_lookahead_min_batch_size=None
        )


def test_decode_mode_cli_rejects_unsupported_config():
    config = Qwen3TTSPipelineConfig(model_path="dummy")
    with pytest.raises(
        typer.BadParameter, match="currently supports only Higgs TTS and MOSS-TTS-Local"
    ):
        apply_decode_mode_cli_overrides(
            config, decode_mode="sync", async_lookahead_min_batch_size=None
        )


def test_decode_mode_cli_absent_is_noop_without_tts_engine_stage():
    # serve() calls this for every model; pipelines with no tts_engine stage
    # (e.g. Qwen3-Omni, Ming) must serve unaffected when decode mode and the
    # advanced async-lookahead threshold are left unspecified.
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="thinker",
                process="pipeline",
                factory="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                terminal=True,
            )
        ],
    )
    result = apply_decode_mode_cli_overrides(
        config, decode_mode=None, async_lookahead_min_batch_size=None
    )
    assert result is config
    assert all(
        "enable_async_decode" not in (stage.factory_args or {})
        for stage in result.stages
    )


def test_async_lookahead_min_batch_size_without_tts_engine_fails_fast():
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="thinker",
                process="pipeline",
                factory="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                terminal=True,
            )
        ],
    )
    with pytest.raises(typer.BadParameter, match="tts_engine"):
        apply_decode_mode_cli_overrides(
            config, decode_mode=None, async_lookahead_min_batch_size=4
        )


def test_generation_server_args_override_updates_server_args():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    apply_generation_server_args_cli_override(
        config,
        max_running_requests=64,
        cuda_graph_max_bs=128,
    )
    args = _tts_engine_args(config)
    assert args["server_args_overrides"]["max_running_requests"] == 64
    assert args["server_args_overrides"]["cuda_graph_max_bs"] == 128


def test_generation_server_args_fill_in_max_running_requests():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    apply_generation_server_args_cli_override(
        config,
        max_running_requests=None,
        cuda_graph_max_bs=128,
    )
    args = _tts_engine_args(config)
    assert args["server_args_overrides"]["max_running_requests"] == 128
    assert args["server_args_overrides"]["cuda_graph_max_bs"] == 128


def test_generation_server_args_fill_in_cuda_graph_max_bs():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    apply_generation_server_args_cli_override(
        config,
        max_running_requests=128,
        cuda_graph_max_bs=None,
    )
    args = _tts_engine_args(config)
    assert args["server_args_overrides"]["max_running_requests"] == 128
    assert args["server_args_overrides"]["cuda_graph_max_bs"] == 128


def test_generation_server_args_override_must_be_positive():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter, match="--max-running-requests"):
        apply_generation_server_args_cli_override(
            config,
            max_running_requests=0,
            cuda_graph_max_bs=None,
        )
    with pytest.raises(typer.BadParameter, match="must be >= 1"):
        apply_generation_server_args_cli_override(
            config,
            max_running_requests=None,
            cuda_graph_max_bs=0,
        )


def test_generation_server_args_override_supports_qwen3_tts_engine():
    config = Qwen3TTSPipelineConfig(model_path="dummy")
    apply_generation_server_args_cli_override(
        config,
        max_running_requests=64,
        cuda_graph_max_bs=64,
    )
    args = _tts_engine_args(config)
    assert args["server_args_overrides"]["max_running_requests"] == 64
    assert args["server_args_overrides"]["cuda_graph_max_bs"] == 64


def test_generation_server_args_override_supports_qwen3_omni_talker():
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    apply_generation_server_args_cli_override(
        config,
        max_running_requests=64,
        cuda_graph_max_bs=64,
    )
    args = _stage_args(config, "talker_ar")
    assert args["server_args_overrides"]["max_running_requests"] == 64
    assert args["server_args_overrides"]["cuda_graph_max_bs"] == 64


def test_generation_server_args_absent_is_noop_without_generation_stage():
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="thinker",
                process="pipeline",
                factory="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                terminal=True,
            )
        ],
    )
    result = apply_generation_server_args_cli_override(
        config,
        max_running_requests=None,
        cuda_graph_max_bs=None,
    )
    assert result is config


def test_generation_server_args_override_without_generation_stage_fails_fast():
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="preprocessing",
                process="pipeline",
                factory="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                terminal=True,
            )
        ],
    )
    with pytest.raises(typer.BadParameter, match="--max-running-requests"):
        apply_generation_server_args_cli_override(
            config,
            max_running_requests=64,
            cuda_graph_max_bs=64,
        )
