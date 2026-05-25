# SPDX-License-Identifier: Apache-2.0
"""Higgs TTS SeedTTS benchmark and cache-hit comparison.

This entrypoint is intentionally thin: normal generation/transcription reuses
``benchmark_tts_seedtts``.  The Higgs-specific part is the cache test, which
compares unique reference audio against a single reused reference audio.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import time
import uuid
from dataclasses import dataclass
from statistics import mean
from typing import Iterable

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import (
    get_wav_duration,
    save_json_results,
    wait_for_service,
)
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    run_tts_seedtts_benchmark,
    run_tts_seedtts_transcribe,
)
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.tasks.tts import build_base_url

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_HIGGS_MODEL = "boson-sglang/higgs-audio-v3-tts-4b-base"
TEXT_PREVIEW_LENGTH = 60


@dataclass
class CacheScenarioResult:
    scenario: str
    outputs: list[RequestResult]
    cold_count: int = 0


def _set_cache_metrics_from_headers(result: RequestResult, headers) -> None:
    cached_tok = headers.get("X-Cached-Tokens")
    cache_hit_rate = headers.get("X-Cache-Hit-Rate")
    if cached_tok is not None:
        result.cached_tokens = int(cached_tok)
    if cache_hit_rate is not None:
        result.cache_hit_rate = float(cache_hit_rate)


def _set_usage_metrics_from_headers(result: RequestResult, headers) -> None:
    prompt_tok = headers.get("X-Prompt-Tokens")
    comp_tok = headers.get("X-Completion-Tokens")
    eng_time = headers.get("X-Engine-Time")
    if prompt_tok is not None:
        result.prompt_tokens = int(prompt_tok)
    if comp_tok is not None:
        result.completion_tokens = int(comp_tok)
    if eng_time is not None:
        result.engine_time_s = float(eng_time)
    _set_cache_metrics_from_headers(result, headers)
    if result.completion_tokens > 0 and result.engine_time_s > 0:
        result.tok_per_s = result.completion_tokens / result.engine_time_s


def _build_cache_test_payload(
    sample: SampleInput, model: str, generation_kwargs: dict
) -> dict:
    payload = {
        "model": model,
        "input": sample.target_text,
        "response_format": "wav",
        "ref_audio": sample.ref_audio,
        "ref_text": sample.ref_text,
    }
    for key, value in generation_kwargs.items():
        if value is not None:
            payload[key] = value
    return payload


def _make_higgs_cache_send_fn(
    model: str,
    api_url: str,
    *,
    save_audio_dir: str,
    generation_kwargs: dict,
):
    async def send_fn(
        session: aiohttp.ClientSession, sample: SampleInput
    ) -> RequestResult:
        result = RequestResult(
            request_id=sample.sample_id,
            text=sample.target_text[:TEXT_PREVIEW_LENGTH],
        )
        payload = _build_cache_test_payload(sample, model, generation_kwargs)
        start_time = time.perf_counter()
        try:
            async with session.post(api_url, json=payload) as response:
                if response.status != 200:
                    result.error = f"HTTP {response.status}: {await response.text()}"
                    return result

                audio_bytes = await response.read()
                result.audio_duration_s = get_wav_duration(audio_bytes)
                if result.audio_duration_s <= 0:
                    result.error = (
                        f"Empty or invalid audio response ({len(audio_bytes)} bytes)"
                    )
                    return result

                result.is_success = True
                result.rtf = (
                    time.perf_counter() - start_time
                ) / result.audio_duration_s
                _set_usage_metrics_from_headers(result, response.headers)
                if audio_bytes:
                    audio_path = os.path.join(save_audio_dir, f"{result.request_id}.wav")
                    with open(audio_path, "wb") as fh:
                        fh.write(audio_bytes)
                    result.wav_path = audio_path
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time
        return result

    return send_fn


def _build_generation_kwargs(args: argparse.Namespace) -> dict:
    return {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }


def _config_from_args(args: argparse.Namespace) -> TtsSeedttsBenchmarkConfig:
    return TtsSeedttsBenchmarkConfig(
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        model=args.model,
        meta=args.meta,
        voice=None,
        voice_clone=True,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=None,
        warmup=args.warmup,
        concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        stream=False,
        disable_tqdm=args.disable_tqdm,
        lang=args.lang,
        device=args.device,
        similarity_checkpoint=None,
    )


def _fixed_ref_samples(
    samples: list[SampleInput],
    *,
    fixed_ref_audio: str | None,
    fixed_ref_text: str | None,
) -> list[SampleInput]:
    if not samples:
        return []
    ref = samples[0]
    ref_audio = fixed_ref_audio or ref.ref_audio
    ref_text = fixed_ref_text if fixed_ref_text is not None else ref.ref_text
    return [
        SampleInput(
            sample_id=f"{sample.sample_id}_same_ref",
            ref_text=ref_text,
            ref_audio=ref_audio,
            target_text=sample.target_text,
        )
        for sample in samples
    ]


async def _run_samples(
    *,
    samples: list[SampleInput],
    api_url: str,
    model: str,
    concurrency: int,
    request_rate: float,
    output_dir: str,
    save_audio_subdir: str,
    disable_tqdm: bool,
    generation_kwargs: dict,
) -> list[RequestResult]:
    save_audio_dir = os.path.abspath(os.path.join(output_dir, save_audio_subdir))
    os.makedirs(save_audio_dir, exist_ok=True)
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=concurrency,
            request_rate=request_rate,
            warmup=0,
            disable_tqdm=disable_tqdm,
        )
    )
    send_fn = _make_higgs_cache_send_fn(
        model,
        api_url,
        save_audio_dir=save_audio_dir,
        generation_kwargs=generation_kwargs,
    )
    return await runner.run(samples, send_fn)


async def _run_one_sample(
    *,
    sample: SampleInput,
    api_url: str,
    model: str,
    output_dir: str,
    save_audio_subdir: str,
    generation_kwargs: dict,
) -> RequestResult:
    save_audio_dir = os.path.abspath(os.path.join(output_dir, save_audio_subdir))
    os.makedirs(save_audio_dir, exist_ok=True)
    send_fn = _make_higgs_cache_send_fn(
        model,
        api_url,
        save_audio_dir=save_audio_dir,
        generation_kwargs=generation_kwargs,
    )
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return await send_fn(session, sample)


def _successful_metric(outputs: Iterable[RequestResult], attr: str) -> list[float]:
    values: list[float] = []
    for output in outputs:
        if not output.is_success:
            continue
        value = float(getattr(output, attr, 0.0) or 0.0)
        if value > 0:
            values.append(value)
    return values


def _mean_or_zero(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def _successful_count(outputs: Iterable[RequestResult]) -> int:
    return sum(1 for output in outputs if output.is_success)


def _successful_metric_including_zero(
    outputs: Iterable[RequestResult], attr: str
) -> list[float]:
    return [
        float(getattr(output, attr, 0.0) or 0.0)
        for output in outputs
        if output.is_success
    ]


def _cache_hit_count(outputs: Iterable[RequestResult]) -> int:
    return sum(
        1
        for output in outputs
        if output.is_success and int(getattr(output, "cached_tokens", 0) or 0) > 0
    )


def _summary(
    *,
    miss_outputs: list[RequestResult],
    hit_outputs: list[RequestResult],
    hit_cold_count: int,
    concurrency: int,
) -> dict:
    hit_measured = hit_outputs[hit_cold_count:]
    miss_latency = _successful_metric(miss_outputs, "latency_s")
    hit_latency = _successful_metric(hit_measured, "latency_s")
    miss_engine = _successful_metric(miss_outputs, "engine_time_s")
    hit_engine = _successful_metric(hit_measured, "engine_time_s")
    miss_cached_tokens = _successful_metric_including_zero(
        miss_outputs, "cached_tokens"
    )
    hit_cached_tokens = _successful_metric_including_zero(
        hit_measured, "cached_tokens"
    )
    miss_cache_hit_rate = _successful_metric_including_zero(
        miss_outputs, "cache_hit_rate"
    )
    hit_cache_hit_rate = _successful_metric_including_zero(
        hit_measured, "cache_hit_rate"
    )

    miss_latency_mean = _mean_or_zero(miss_latency)
    hit_latency_mean = _mean_or_zero(hit_latency)
    miss_engine_mean = _mean_or_zero(miss_engine)
    hit_engine_mean = _mean_or_zero(hit_engine)
    hit_expected_cache_hits = _successful_count(hit_measured)
    miss_cache_hits = _cache_hit_count(miss_outputs)
    hit_cache_hits = _cache_hit_count(hit_measured)

    return {
        "concurrency": concurrency,
        "miss_samples": len(miss_outputs),
        "hit_samples": len(hit_measured),
        "hit_cold_excluded": hit_cold_count,
        "miss_failed": sum(not o.is_success for o in miss_outputs),
        "hit_failed": sum(not o.is_success for o in hit_measured),
        "miss_expected_cache_hits": 0,
        "hit_expected_cache_hits": hit_expected_cache_hits,
        "miss_expected_request_cache_hit_rate": 0.0,
        "hit_expected_request_cache_hit_rate": round(
            hit_expected_cache_hits / len(hit_measured), 6
        )
        if hit_measured
        else 0.0,
        "miss_cache_hits": miss_cache_hits,
        "hit_cache_hits": hit_cache_hits,
        "miss_request_cache_hit_rate": round(
            miss_cache_hits / len(miss_outputs), 6
        )
        if miss_outputs
        else 0.0,
        "hit_request_cache_hit_rate": round(hit_cache_hits / len(hit_measured), 6)
        if hit_measured
        else 0.0,
        "miss_cached_tokens_mean": _mean_or_zero(miss_cached_tokens),
        "hit_cached_tokens_mean": _mean_or_zero(hit_cached_tokens),
        "miss_token_cache_hit_rate_mean": _mean_or_zero(miss_cache_hit_rate),
        "hit_token_cache_hit_rate_mean": _mean_or_zero(hit_cache_hit_rate),
        "miss_latency_mean_s": miss_latency_mean,
        "hit_latency_mean_s": hit_latency_mean,
        "latency_saved_s": round(miss_latency_mean - hit_latency_mean, 6)
        if miss_latency_mean and hit_latency_mean
        else 0.0,
        "latency_speedup_ratio": round(miss_latency_mean / hit_latency_mean, 4)
        if hit_latency_mean
        else None,
        "miss_engine_time_mean_s": miss_engine_mean,
        "hit_engine_time_mean_s": hit_engine_mean,
        "engine_time_saved_s": round(miss_engine_mean - hit_engine_mean, 6)
        if miss_engine_mean and hit_engine_mean
        else 0.0,
        "engine_time_speedup_ratio": round(miss_engine_mean / hit_engine_mean, 4)
        if hit_engine_mean
        else None,
    }


def _row(scenario: str, index: int, output: RequestResult, *, is_cold: bool) -> dict:
    return {
        "scenario": scenario,
        "request_index": index,
        "id": output.request_id,
        "is_cold": is_cold,
        "expected_cache_hit": scenario == "same_ref_hit" and not is_cold,
        "is_success": output.is_success,
        "latency_s": round(output.latency_s, 6),
        "engine_time_s": round(output.engine_time_s, 6)
        if output.engine_time_s > 0
        else None,
        "audio_duration_s": round(output.audio_duration_s, 6),
        "rtf": round(output.rtf, 6) if output.rtf < float("inf") else None,
        "prompt_tokens": output.prompt_tokens or None,
        "completion_tokens": output.completion_tokens or None,
        "cached_tokens": getattr(output, "cached_tokens", 0),
        "cache_hit_rate": round(float(getattr(output, "cache_hit_rate", 0.0)), 6),
        "tok_per_s": round(output.tok_per_s, 6) if output.tok_per_s > 0 else None,
        "wav_path": output.wav_path or None,
        "error": output.error or None,
    }


def _save_cache_results(
    *,
    output_dir: str,
    summary: dict,
    scenarios: list[CacheScenarioResult],
    config: dict,
) -> None:
    rows: list[dict] = []
    for scenario in scenarios:
        for i, output in enumerate(scenario.outputs):
            rows.append(
                _row(
                    scenario.scenario,
                    i,
                    output,
                    is_cold=i < scenario.cold_count,
                )
            )

    save_json_results(
        {"summary": summary, "config": config, "per_request": rows},
        output_dir,
        "cache_results.json",
    )

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "cache_results.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Cache results saved to %s", output_dir)


async def run_cache_test(args: argparse.Namespace) -> dict:
    if args.cache_test_samples <= 1:
        raise ValueError("--cache-test-samples must be greater than 1")
    loaded_samples = load_seedtts_samples(args.meta, args.cache_test_samples + 2)
    required_samples = args.cache_test_samples + 2
    if len(loaded_samples) < required_samples:
        raise ValueError(
            f"Need at least {required_samples} SeedTTS samples, got "
            f"{len(loaded_samples)}"
        )
    fixed_ref_sample = loaded_samples[0]
    miss_warmup_sample = loaded_samples[1]
    samples = loaded_samples[2:]

    base_url = build_base_url(_config_from_args(args))
    api_url = f"{base_url}/v1/audio/speech"
    generation_kwargs = _build_generation_kwargs(args)

    await _run_one_sample(
        sample=miss_warmup_sample,
        api_url=api_url,
        model=args.model,
        output_dir=args.output_dir,
        save_audio_subdir="cache_miss_warmup_audio",
        generation_kwargs=generation_kwargs,
    )

    miss_outputs = await _run_samples(
        samples=samples,
        api_url=api_url,
        model=args.model,
        concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        output_dir=args.output_dir,
        save_audio_subdir="cache_miss_audio",
        disable_tqdm=args.disable_tqdm,
        generation_kwargs=generation_kwargs,
    )

    same_ref = _fixed_ref_samples(
        samples,
        fixed_ref_audio=args.fixed_ref_audio or fixed_ref_sample.ref_audio,
        fixed_ref_text=args.fixed_ref_text
        if args.fixed_ref_text is not None
        else fixed_ref_sample.ref_text,
    )
    cold = await _run_one_sample(
        sample=same_ref[0],
        api_url=api_url,
        model=args.model,
        output_dir=args.output_dir,
        save_audio_subdir="cache_hit_audio",
        generation_kwargs=generation_kwargs,
    )
    hit_rest = await _run_samples(
        samples=same_ref[1:],
        api_url=api_url,
        model=args.model,
        concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        output_dir=args.output_dir,
        save_audio_subdir="cache_hit_audio",
        disable_tqdm=args.disable_tqdm,
        generation_kwargs=generation_kwargs,
    )
    hit_outputs = [cold, *hit_rest]

    summary = _summary(
        miss_outputs=miss_outputs,
        hit_outputs=hit_outputs,
        hit_cold_count=1,
        concurrency=args.max_concurrency,
    )
    config = {
        "model": args.model,
        "base_url": base_url,
        "meta": args.meta,
        "cache_test_samples": args.cache_test_samples,
        "concurrency": args.max_concurrency,
        "request_rate": args.request_rate,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "fixed_ref_audio": args.fixed_ref_audio or fixed_ref_sample.ref_audio,
        "fixed_ref_text": args.fixed_ref_text or fixed_ref_sample.ref_text,
        "miss_warmup_sample_id": miss_warmup_sample.sample_id,
    }
    _save_cache_results(
        output_dir=args.output_dir,
        summary=summary,
        config=config,
        scenarios=[
            CacheScenarioResult("unique_ref_miss", miss_outputs, cold_count=0),
            CacheScenarioResult("same_ref_hit", hit_outputs, cold_count=1),
        ],
    )
    print_cache_summary(summary)
    return {"summary": summary, "config": config}


def print_cache_summary(summary: dict) -> None:
    print("\n================ Higgs TTS Cache Test ================")
    print(f"  Concurrency:                 {summary['concurrency']}")
    print(f"  Miss samples:                {summary['miss_samples']}")
    print(f"  Hit samples:                 {summary['hit_samples']}")
    print(f"  Hit cold excluded:           {summary['hit_cold_excluded']}")
    print(f"  Miss expected cache hits:    {summary['miss_expected_cache_hits']}")
    print(f"  Hit expected cache hits:     {summary['hit_expected_cache_hits']}")
    print(
        "  Miss expected hit rate:      "
        f"{summary['miss_expected_request_cache_hit_rate']}"
    )
    print(
        "  Hit expected hit rate:       "
        f"{summary['hit_expected_request_cache_hit_rate']}"
    )
    print(f"  Miss actual cache hits:      {summary['miss_cache_hits']}")
    print(f"  Hit actual cache hits:       {summary['hit_cache_hits']}")
    print(f"  Miss actual hit rate:        {summary['miss_request_cache_hit_rate']}")
    print(f"  Hit actual hit rate:         {summary['hit_request_cache_hit_rate']}")
    print(f"  Miss cached tokens mean:     {summary['miss_cached_tokens_mean']}")
    print(f"  Hit cached tokens mean:      {summary['hit_cached_tokens_mean']}")
    print(f"  Miss token hit rate mean:    {summary['miss_token_cache_hit_rate_mean']}")
    print(f"  Hit token hit rate mean:     {summary['hit_token_cache_hit_rate_mean']}")
    print(f"  Miss latency mean (s):       {summary['miss_latency_mean_s']}")
    print(f"  Hit latency mean (s):        {summary['hit_latency_mean_s']}")
    print(f"  Latency saved (s):           {summary['latency_saved_s']}")
    print(f"  Latency speedup:             {summary['latency_speedup_ratio']}")
    print(f"  Miss engine mean (s):        {summary['miss_engine_time_mean_s']}")
    print(f"  Hit engine mean (s):         {summary['hit_engine_time_mean_s']}")
    print(f"  Engine time saved (s):       {summary['engine_time_saved_s']}")
    print(f"  Engine time speedup:         {summary['engine_time_speedup_ratio']}")
    print("======================================================\n")


async def run_generate(args: argparse.Namespace) -> dict:
    config = _config_from_args(args)
    results = await run_tts_seedtts_benchmark(config)
    print_speed_summary(
        results["summary"],
        config.model,
        concurrency=config.concurrency,
        title="Higgs TTS Speed Benchmark Result",
    )
    return results


def _default_profile_template(profile_output_dir: str, run_id: str) -> str:
    return os.path.abspath(os.path.join(profile_output_dir, run_id, "{stage}", "trace"))


def _default_profile_event_dir(profile_output_dir: str, run_id: str) -> str:
    return os.path.abspath(os.path.join(profile_output_dir, run_id, "events"))


async def _post_profile_control(
    *,
    base_url: str,
    endpoint: str,
    body: dict,
) -> dict:
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{base_url}/{endpoint}", json=body) as response:
            text = await response.text()
            if response.status != 200:
                raise RuntimeError(
                    f"{endpoint} failed with HTTP {response.status}: {text}"
                )
            try:
                return await response.json()
            except aiohttp.ContentTypeError:
                return {"raw": text}


async def _start_profile(args: argparse.Namespace) -> tuple[str, str] | None:
    if not args.profile:
        return None
    run_id = args.profile_run_id or f"higgs_tts_{uuid.uuid4().hex[:8]}"
    base_url = build_base_url(_config_from_args(args))
    body: dict = {"run_id": run_id}
    if args.profile_mode == "request":
        endpoint = "start_request_profile"
        if args.profile_output_dir:
            body["event_dir"] = _default_profile_event_dir(
                args.profile_output_dir, run_id
            )
    else:
        endpoint = "start_profile"
        if args.profile_output_dir:
            body["trace_path_template"] = _default_profile_template(
                args.profile_output_dir, run_id
            )
            body["event_dir"] = _default_profile_event_dir(
                args.profile_output_dir, run_id
            )
        body["enable_torch"] = True
    result = await _post_profile_control(
        base_url=base_url,
        endpoint=endpoint,
        body=body,
    )
    logger.info("Started profiler: %s", result)
    return run_id, endpoint


async def _stop_profile(
    args: argparse.Namespace, started: tuple[str, str] | None
) -> None:
    if started is None:
        return
    run_id, start_endpoint = started
    stop_endpoint = (
        "stop_request_profile"
        if start_endpoint == "start_request_profile"
        else "stop_profile"
    )
    body: dict = {"run_id": run_id}
    base_url = build_base_url(_config_from_args(args))
    result = await _post_profile_control(
        base_url=base_url,
        endpoint=stop_endpoint,
        body=body,
    )
    logger.info("Stopped profiler: %s", result)

async def _run_with_optional_profile(args: argparse.Namespace, run_coro_factory) -> dict:
    started = await _start_profile(args)
    try:
        return await run_coro_factory()
    finally:
        await _stop_profile(args, started)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Higgs TTS SeedTTS benchmark and cache-hit comparison."
    )
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default=DEFAULT_HIGGS_MODEL)
    parser.add_argument(
        "--meta",
        "--testset",
        dest="meta",
        type=str,
        default="seedtts_testset/en/meta.lst",
    )
    parser.add_argument("--output-dir", type=str, default="results/higgs_tts")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--concurrency",
        "--max-concurrency",
        dest="max_concurrency",
        type=int,
        default=1,
    )
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--lang", type=str, choices=["en", "zh"], default="en")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--server-timeout", type=int, default=1200)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture profiler output during generation/cache test.",
    )
    parser.add_argument(
        "--profile-mode",
        choices=("torch", "request"),
        default="torch",
        help=(
            "Profiler mode: torch captures Torch traces plus request events; "
            "request captures lower-overhead stage events only."
        ),
    )
    parser.add_argument(
        "--profile-output-dir",
        type=str,
        default=None,
        help=(
            "Directory for profiler output. Torch mode writes traces and "
            "stage events; request mode writes stage events only."
        ),
    )
    parser.add_argument(
        "--profile-run-id",
        type=str,
        default=None,
        help="Optional profiler run id. Defaults to a higgs_tts_<uuid> value.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate-only", action="store_true")
    mode.add_argument("--transcribe-only", action="store_true")
    mode.add_argument("--cache-test", action="store_true")

    parser.add_argument(
        "--cache-test-samples",
        type=int,
        default=50,
        help=(
            "Number of target requests per cache scenario. The loader reads one "
            "extra SeedTTS row as the fixed reference for same-ref cache hits, "
            "and one more row as the unique-ref miss warmup request."
        ),
    )
    parser.add_argument(
        "--fixed-ref-audio",
        type=str,
        default=None,
        help="Optional fixed reference audio for same-ref cache-hit scenario.",
    )
    parser.add_argument(
        "--fixed-ref-text",
        type=str,
        default=None,
        help="Optional transcript for --fixed-ref-audio.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.transcribe_only:
        run_tts_seedtts_transcribe(_config_from_args(args))
        return

    wait_for_service(
        build_base_url(_config_from_args(args)), timeout=args.server_timeout
    )

    if args.cache_test:
        asyncio.run(_run_with_optional_profile(args, lambda: run_cache_test(args)))
        return

    asyncio.run(_run_with_optional_profile(args, lambda: run_generate(args)))


if __name__ == "__main__":
    main()
