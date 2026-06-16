# SPDX-License-Identifier: Apache-2.0
"""CLI overrides for SGLang generation-stage server args."""

from __future__ import annotations

import pytest
import typer

from sglang_omni.cli.serve import apply_generation_server_args_cli_override
from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.moss_tts.config import MossTTSPipelineConfig
from sglang_omni.models.moss_tts_local.config import MossTTSLocalPipelineConfig
from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.models.voxtral_tts.config import VoxtralTTSPipelineConfig
from sglang_omni.models.voxtral_tts.pipeline.next_stage import GENERATION_STAGE


def _stage_args(config: PipelineConfig, stage_name: str) -> dict[str, object]:
    stage = next(s for s in config.stages if s.name == stage_name)
    return dict(stage.factory_args or {})


class ExplicitGenerationPipelineConfig(PipelineConfig):
    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "custom_generation"}


def test_generation_server_args_use_explicit_role_map():
    config = ExplicitGenerationPipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="tts_engine",
                process="pipeline",
                factory="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                next="custom_generation",
            ),
            StageConfig(
                name="custom_generation",
                process="pipeline",
                factory="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                terminal=True,
            ),
        ],
    )
    apply_generation_server_args_cli_override(
        config,
        max_running_requests=32,
        cuda_graph_max_bs=64,
    )

    assert "server_args_overrides" not in _stage_args(config, "tts_engine")
    overrides = _stage_args(config, "custom_generation")["server_args_overrides"]
    assert overrides["max_running_requests"] == 32
    assert overrides["cuda_graph_max_bs"] == 64


def test_generation_server_args_fill_in_max_running_requests():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    apply_generation_server_args_cli_override(
        config,
        max_running_requests=None,
        cuda_graph_max_bs=128,
    )
    overrides = _stage_args(config, "tts_engine")["server_args_overrides"]
    assert overrides["max_running_requests"] == 128
    assert overrides["cuda_graph_max_bs"] == 128


def test_generation_server_args_fill_in_cuda_graph_max_bs():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    apply_generation_server_args_cli_override(
        config,
        max_running_requests=128,
        cuda_graph_max_bs=None,
    )
    overrides = _stage_args(config, "tts_engine")["server_args_overrides"]
    assert overrides["max_running_requests"] == 128
    assert overrides["cuda_graph_max_bs"] == 128


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


@pytest.mark.parametrize(
    "config_cls",
    [
        HiggsTtsPipelineConfig,
        Qwen3TTSPipelineConfig,
        MossTTSPipelineConfig,
        MossTTSLocalPipelineConfig,
        S2ProPipelineConfig,
    ],
)
def test_generation_server_args_support_migrated_tts_configs(config_cls):
    config = config_cls(model_path="dummy")
    apply_generation_server_args_cli_override(
        config,
        max_running_requests=64,
        cuda_graph_max_bs=64,
    )
    overrides = _stage_args(config, "tts_engine")["server_args_overrides"]
    assert overrides["max_running_requests"] == 64
    assert overrides["cuda_graph_max_bs"] == 64


def test_generation_server_args_support_qwen3_omni_speech():
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    apply_generation_server_args_cli_override(
        config,
        max_running_requests=64,
        cuda_graph_max_bs=64,
    )
    overrides = _stage_args(config, "talker_ar")["server_args_overrides"]
    assert overrides["max_running_requests"] == 64
    assert overrides["cuda_graph_max_bs"] == 64


def test_generation_server_args_support_voxtral_tts():
    config = VoxtralTTSPipelineConfig(model_path="dummy")
    apply_generation_server_args_cli_override(
        config,
        max_running_requests=64,
        cuda_graph_max_bs=64,
    )
    args = _stage_args(config, GENERATION_STAGE)
    overrides = args["server_args_overrides"]
    assert overrides["max_running_requests"] == 64
    assert overrides["cuda_graph_max_bs"] == 64


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
                name="tts_engine",
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
