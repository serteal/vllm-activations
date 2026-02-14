# SPDX-License-Identifier: Apache-2.0
"""Activation collection engine built on vLLM.

Provides a thin wrapper around vLLM's LLM class that captures intermediate
layer activations during prefill.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch

from vllm.entrypoints.llm import LLM
from vllm.sampling_params import RequestOutputKind, SamplingParams


@dataclass
class ActivationResult:
    """Container for collected activations.

    Attributes:
        activations: Mapping from layer index to list of tensors.
            Each tensor has shape [seq_len, hidden_dim] for that sequence.
        num_tokens: Number of tokens per input sequence.
    """

    activations: dict[int, list[torch.Tensor]]
    num_tokens: list[int] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def stack(self, layer: int) -> torch.Tensor:
        """Stack activations for a single layer into a padded batch tensor.

        Returns tensor of shape [batch, max_seq_len, hidden_dim].
        """
        tensors = self.activations[layer]
        if not tensors:
            raise ValueError("No activations collected")
        max_len = max(t.shape[0] for t in tensors)
        hidden = tensors[0].shape[-1]
        out = torch.zeros(len(tensors), max_len, hidden, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            out[i, : t.shape[0]] = t
        return out

    def mask(self) -> torch.Tensor:
        """Return boolean mask [batch, max_seq_len] for padded positions."""
        if not self.num_tokens:
            raise ValueError("No token counts recorded")
        max_len = max(self.num_tokens)
        mask = torch.zeros(len(self.num_tokens), max_len, dtype=torch.bool)
        for i, n in enumerate(self.num_tokens):
            mask[i, :n] = True
        return mask


@dataclass
class FlatActivationResult:
    """Flat in-memory activation payload.

    Attributes:
        layers: Mapping layer payload. Payloads may be either:
            - materialized: {"values": [total_tokens, hidden],
              "offsets": [num_requests + 1]}
            - chunked: {"__chunked__": True, "values": [chunk tensors],
              "lengths": [per-chunk length tensors], "offsets": [...]}
        num_tokens: Prompt token count per request in returned order.
        req_ids: External request ids in returned order.
    """

    layers: dict[int, dict[str, Any]]
    num_tokens: list[int] = field(default_factory=list)
    req_ids: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def to_activation_result(self) -> ActivationResult:
        """Materialize per-request tensors from flat representation."""
        activations: dict[int, list[torch.Tensor]] = {}
        for layer, payload in self.layers.items():
            seq_tensors: list[torch.Tensor] = []

            if payload.get("__chunked__", False):
                value_chunks = payload.get("values", [])
                length_chunks = payload.get("lengths", [])
                for values, lengths in zip(value_chunks, length_chunks):
                    if isinstance(lengths, torch.Tensor):
                        lengths_list = lengths.tolist()
                    else:
                        lengths_list = [int(x) for x in lengths]
                    hidden = int(values.shape[1]) if values.ndim > 1 else 0
                    offset = 0
                    for n in lengths_list:
                        n_i = int(n)
                        if n_i <= 0:
                            seq_tensors.append(values.new_empty((0, hidden)))
                            continue
                        seq_tensors.append(values[offset : offset + n_i])
                        offset += n_i
                if len(seq_tensors) < len(self.num_tokens):
                    if value_chunks:
                        first = value_chunks[0]
                        hidden = int(first.shape[1]) if first.ndim > 1 else 0
                        empty = first.new_empty((0, hidden))
                    else:
                        empty = torch.empty((0, 0), dtype=torch.float32)
                    seq_tensors.extend([empty] * (len(self.num_tokens) - len(seq_tensors)))
                elif len(seq_tensors) > len(self.num_tokens):
                    seq_tensors = seq_tensors[: len(self.num_tokens)]
                for i in range(len(seq_tensors)):
                    seq_tensors[i] = seq_tensors[i][: self.num_tokens[i]]
            else:
                values = payload["values"]
                offsets = payload["offsets"]
                offsets_list = offsets.tolist()
                for i in range(len(self.num_tokens)):
                    start = int(offsets_list[i])
                    end = int(offsets_list[i + 1])
                    seq_tensors.append(values[start:end][: self.num_tokens[i]])
            activations[layer] = seq_tensors
        return ActivationResult(
            activations=activations,
            num_tokens=list(self.num_tokens),
            timings=dict(self.timings),
        )


class ActivationEngine:
    """Collect intermediate layer activations using vLLM's inference engine.

    Args:
        model: HuggingFace model ID or local path.
        layers: Layer indices to capture activations from.
        hook_point: Where to capture: "post_block".
        dtype: Model dtype ("auto", "float16", "bfloat16", "float32").
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory to use.
        max_num_batched_tokens: Scheduler token budget per step.
        max_num_seqs: Optional max concurrent requests in scheduler.
        max_num_partial_prefills: Optional partial-prefill concurrency cap.
        max_long_partial_prefills: Optional long-prefill concurrency cap.
        long_prefill_token_threshold: Optional threshold for "long prefill".
        enable_chunked_prefill: Optionally enable/disable chunked prefill.
        compilation_config: Optional vLLM compilation/cudagraph config override.
        prefill_only: End requests immediately after prefill (no decode outputs).
        staged_export: Enable staged activation export in single-process mode
            to overlap GPU->CPU copy of batch N with compute of batch N+1.
        static_shape_bucketing: Group same-length requests together when batching.
        prefill_cudagraph: Prefer non-eager execution so static-shape buckets can
            use CUDA graph paths.
        activation_export_device: Activation payload export device ("cpu" or
            "cuda"). "cuda" avoids GPU->CPU copies in single-process mode.
        text_only: Disable multimodal prompt items by default (image limit=0).
            This avoids unnecessary multimodal profiling overhead/OOM when using
            ActivationEngine for text activation collection.
        **kwargs: Additional arguments passed to ``vllm.LLM``.
    """

    def __init__(
        self,
        model: str,
        layers: list[int],
        hook_point: str = "post_block",
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = 4096,
        enforce_eager: bool = False,
        max_num_batched_tokens: int = 65536,
        max_num_seqs: int | None = None,
        max_num_partial_prefills: int | None = None,
        max_long_partial_prefills: int | None = None,
        long_prefill_token_threshold: int | None = None,
        enable_chunked_prefill: bool | None = None,
        compilation_config: dict[str, Any] | None = None,
        pool_mode: str | None = None,
        activation_only: bool = True,
        tp_rank0_only: bool = True,
        flat_output: bool = True,
        prefill_only: bool = True,
        staged_export: bool = True,
        static_shape_bucketing: bool = True,
        prefill_cudagraph: bool = True,
        activation_export_device: str = "cuda",
        text_only: bool = True,
        **kwargs: Any,
    ):
        self.layers = sorted(layers)
        self.hook_point = hook_point
        self.pool_mode = pool_mode
        self.activation_only = activation_only
        self.tp_rank0_only = tp_rank0_only
        self.flat_output = flat_output
        self.prefill_only = prefill_only
        self.staged_export = staged_export
        self.static_shape_bucketing = static_shape_bucketing
        self.prefill_cudagraph = prefill_cudagraph
        if activation_export_device not in ("cpu", "cuda"):
            raise ValueError(
                "activation_export_device must be one of {'cpu', 'cuda'}"
            )
        self.activation_export_device = activation_export_device
        self.text_only = text_only
        self._active_pool_mode: str | None = None
        self._closed = False

        # Disable multiprocessing for activation-heavy workloads. In-process mode
        # avoids serialization overhead on large tensor payloads.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        # Set capture layers before model init so capture branches are traced.
        os.environ["_VLLM_ACTIVATION_CAPTURE_LAYERS"] = ",".join(
            str(l) for l in self.layers
        )

        if prefill_cudagraph and enforce_eager:
            enforce_eager = False

        llm_kwargs: dict[str, Any] = {
            "model": model,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "max_num_batched_tokens": max_num_batched_tokens,
            # Activation fast path requires sync scheduling.
            "async_scheduling": False,
        }
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = max_num_seqs
        if max_num_partial_prefills is not None:
            llm_kwargs["max_num_partial_prefills"] = max_num_partial_prefills
        if max_long_partial_prefills is not None:
            llm_kwargs["max_long_partial_prefills"] = max_long_partial_prefills
        if long_prefill_token_threshold is not None:
            llm_kwargs["long_prefill_token_threshold"] = long_prefill_token_threshold
        if enable_chunked_prefill is not None:
            llm_kwargs["enable_chunked_prefill"] = enable_chunked_prefill
        if compilation_config is not None:
            llm_kwargs["compilation_config"] = compilation_config
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        if text_only and "limit_mm_per_prompt" not in kwargs:
            kwargs["limit_mm_per_prompt"] = {"image": 0}
        llm_kwargs.update(kwargs)

        self.llm = LLM(**llm_kwargs)
        self._local_worker = self._resolve_local_worker()
        if self.activation_export_device != "cpu" and self._local_worker is None:
            # Cross-process RPC payloads are CPU-oriented; keep compatibility.
            self.activation_export_device = "cpu"

        # Register capture layers.
        self._register_capture(self.pool_mode)

    def _resolve_local_worker(self):
        """Return direct local worker handle in single-process mode if available."""
        llm_engine = getattr(self.llm, "llm_engine", None)
        if llm_engine is None:
            return None
        model_executor = getattr(llm_engine, "model_executor", None)
        if model_executor is None:
            return None
        driver_worker = getattr(model_executor, "driver_worker", None)
        if driver_worker is None:
            return None
        required = (
            "register_activation_capture",
            "get_captured_activations",
            "remove_activation_capture",
        )
        if all(hasattr(driver_worker, name) for name in required):
            return driver_worker
        return None

    def _collective_rpc(self, method: str, **kwargs) -> Any:
        """Call a method on all workers via model executor."""
        return self.llm.llm_engine.model_executor.collective_rpc(method, **kwargs)

    def _worker_call(self, method: str, *args, **kwargs) -> list[Any]:
        """Call worker utility directly in uniproc mode, else fallback to RPC."""
        worker = self._local_worker
        if worker is not None and hasattr(worker, method):
            fn = getattr(worker, method)
            return [fn(*args, **kwargs)]
        rpc_kwargs = {"args": args} if args else {}
        if kwargs:
            rpc_kwargs["kwargs"] = kwargs
        return self._collective_rpc(method, **rpc_kwargs)

    @property
    def tokenizer(self):
        return self.llm.get_tokenizer()

    def _register_capture(self, pool_mode: str | None) -> None:
        self._worker_call(
            "register_activation_capture",
            self.layers,
            self.hook_point,
            pool_mode,
            self.activation_only,
            self.tp_rank0_only,
            self.flat_output,
            self.activation_export_device == "cpu",
        )
        self._active_pool_mode = pool_mode

    def _make_sampling_params(self) -> SamplingParams:
        extra_args = {"_activation_prefill_only": True} if self.prefill_only else None
        return SamplingParams(
            max_tokens=1,
            temperature=0.0,
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
            extra_args=extra_args,
        )

    @staticmethod
    def _request_prompt_len(request) -> int:
        if request.prompt_token_ids is not None:
            return len(request.prompt_token_ids)
        if request.prompt_embeds is not None:
            return int(request.prompt_embeds.shape[0])
        return 0

    def _run_prefill_batch(
        self,
        inputs: list,
        sampling_params: SamplingParams,
        include_timings: bool,
    ) -> tuple[list[str], list[int], dict[str, float]]:
        """Run one prefill batch and return (req_ids_in_order, prompt_lens, timings)."""
        timings: dict[str, float] = {}

        if not self.prefill_only:
            t0 = time.perf_counter()
            outputs = self.llm.generate(inputs, sampling_params, use_tqdm=False)
            t1 = time.perf_counter()
            if include_timings:
                timings["model_forward_s"] = t1 - t0
            req_ids = [o.request_id for o in outputs]
            num_tokens = [len(o.prompt_token_ids) for o in outputs]
            return req_ids, num_tokens, timings

        llm_engine = self.llm.llm_engine
        input_processor = llm_engine.input_processor
        engine_core = llm_engine.engine_core
        supported_tasks = llm_engine.get_supported_tasks()

        req_ids: list[str] = []
        num_tokens: list[int] = []
        pending_internal: set[str] = set()

        t0 = time.perf_counter()
        for prompt in inputs:
            external_req_id = str(next(self.llm.request_counter))
            request = input_processor.process_inputs(
                external_req_id,
                prompt,
                sampling_params,
                supported_tasks=supported_tasks,
            )
            input_processor.assign_request_id(request)
            # Activation payload is keyed by external request ID.
            req_ids.append(request.external_req_id or request.request_id)
            num_tokens.append(self._request_prompt_len(request))
            pending_internal.add(request.request_id)
            engine_core.add_request(request)

        # Drive EngineCore directly: no output_processor, no detokenization.
        # Requests are marked finished in scheduler prefill-only mode.
        max_steps = max(8, len(inputs) * 8)
        steps = 0
        while pending_internal:
            outputs = engine_core.get_output()
            for output in outputs.outputs:
                if output.finished:
                    pending_internal.discard(output.request_id)
            steps += 1
            if steps > max_steps:
                raise RuntimeError(
                    "Prefill-only activation run exceeded expected step budget; "
                    "pending requests did not finish"
                )

        t1 = time.perf_counter()
        if include_timings:
            timings["model_forward_s"] = t1 - t0
        return req_ids, num_tokens, timings

    def _assemble_activations(
        self,
        payload: Any,
        req_ids: list[str],
        num_tokens: list[int],
    ) -> dict[int, list[torch.Tensor]]:
        activations: dict[int, list[torch.Tensor]] = {l: [] for l in self.layers}

        if not isinstance(payload, dict):
            for layer in self.layers:
                activations[layer] = [torch.zeros(0) for _ in req_ids]
            return activations

        if payload.get("__flat__") is True:
            payload_req_ids = payload.get("req_ids", [])
            layer_payloads = payload.get("layers", {})
            if payload_req_ids == req_ids:
                req_positions = list(range(len(req_ids)))
            else:
                req_index = {rid: i for i, rid in enumerate(payload_req_ids)}
                req_positions = [req_index.get(rid, -1) for rid in req_ids]

            for layer in self.layers:
                layer_payload = layer_payloads.get(layer)
                if layer_payload is None:
                    activations[layer] = [torch.zeros(0) for _ in req_ids]
                    continue

                values = layer_payload["values"]
                offsets = layer_payload["offsets"]
                offsets_list = offsets.tolist()
                layer_out: list[torch.Tensor] = []
                for i, flat_idx in enumerate(req_positions):
                    if flat_idx < 0:
                        layer_out.append(torch.zeros(0))
                        continue
                    start = int(offsets_list[flat_idx])
                    end = int(offsets_list[flat_idx + 1])
                    layer_out.append(values[start:end][: num_tokens[i]])
                activations[layer] = layer_out
            return activations

        # Legacy per-request payload fallback.
        for i, req_id in enumerate(req_ids):
            req_acts = payload.get(req_id, {})
            prompt_len = num_tokens[i]
            for layer in self.layers:
                if layer in req_acts:
                    activations[layer].append(req_acts[layer][:prompt_len])
                else:
                    activations[layer].append(torch.zeros(0))
        return activations

    def _ordered_indices(
        self,
        token_ids: list[list[int]],
        batch_size: int | None,
        batch_token_budget: int | None,
        sort_by_length: bool,
    ) -> list[int] | None:
        if batch_size is None and batch_token_budget is None:
            return None

        if self.static_shape_bucketing:
            # Group by exact sequence length first, then process long -> short.
            buckets: dict[int, list[int]] = {}
            for i, ids in enumerate(token_ids):
                buckets.setdefault(len(ids), []).append(i)
            ordered: list[int] = []
            for seq_len in sorted(buckets.keys(), reverse=True):
                ordered.extend(buckets[seq_len])
            return ordered

        if sort_by_length:
            return sorted(range(len(token_ids)), key=lambda i: len(token_ids[i]), reverse=True)

        return None

    @staticmethod
    def _build_batch_ranges(
        total_inputs: int,
        batch_size: int | None,
        batch_token_budget: int | None,
        token_lens: list[int] | None,
    ) -> list[tuple[int, int]]:
        if total_inputs == 0:
            return []

        if batch_token_budget is None:
            bs = total_inputs if batch_size is None else batch_size
            return [
                (start, min(start + bs, total_inputs))
                for start in range(0, total_inputs, bs)
            ]

        if token_lens is None:
            raise ValueError(
                "token_lens must be provided when batch_token_budget is set"
            )

        ranges: list[tuple[int, int]] = []
        start = 0
        cur_tokens = 0
        cur_count = 0
        for i, n_tok in enumerate(token_lens):
            # Always allow at least one sequence in each batch, even if it
            # exceeds budget.
            over_budget = cur_count > 0 and (cur_tokens + n_tok > batch_token_budget)
            over_size = batch_size is not None and cur_count >= batch_size
            if over_budget or over_size:
                ranges.append((start, i))
                start = i
                cur_tokens = 0
                cur_count = 0
            cur_tokens += n_tok
            cur_count += 1
        ranges.append((start, total_inputs))
        return ranges

    def collect(
        self,
        prompts: list[str] | None = None,
        token_ids: list[list[int]] | None = None,
        batch_size: int | None = None,
        batch_token_budget: int | None = None,
        pool_mode: str | None = None,
        include_timings: bool = False,
        sort_by_length: bool = True,
    ) -> ActivationResult:
        """Collect activations from the model."""
        if self._closed:
            raise RuntimeError("ActivationEngine is closed")
        if prompts is None and token_ids is None:
            raise ValueError("Provide either prompts or token_ids")
        if prompts is not None and token_ids is not None:
            raise ValueError("Provide only one of prompts or token_ids")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_token_budget is not None and batch_token_budget <= 0:
            raise ValueError("batch_token_budget must be positive")
        if batch_token_budget is not None and token_ids is None:
            raise ValueError("batch_token_budget requires token_ids input")

        if token_ids is not None:
            inputs: list = [{"prompt_token_ids": ids} for ids in token_ids]
        else:
            inputs = prompts  # type: ignore[assignment]

        effective_pool_mode = self.pool_mode if pool_mode is None else pool_mode
        if effective_pool_mode not in (None, "mean", "last_token"):
            raise ValueError(
                f"Unsupported pool_mode={effective_pool_mode!r}. "
                "Expected one of: None, 'mean', 'last_token'."
            )
        if self._active_pool_mode != effective_pool_mode:
            self._register_capture(effective_pool_mode)

        restore_order: list[int] | None = None
        ordered_token_lens: list[int] | None = None
        if token_ids is not None:
            ordered = self._ordered_indices(
                token_ids,
                batch_size,
                batch_token_budget,
                sort_by_length,
            )
            if ordered is not None:
                inputs = [inputs[i] for i in ordered]
                restore_order = ordered
                ordered_token_lens = [len(token_ids[i]) for i in ordered]
            else:
                ordered_token_lens = [len(ids) for ids in token_ids]

        sampling_params = self._make_sampling_params()
        result = self._collect_batches(
            inputs,
            batch_size=batch_size,
            batch_token_budget=batch_token_budget,
            token_lens=ordered_token_lens,
            sampling_params=sampling_params,
            include_timings=include_timings,
        )

        if restore_order is None:
            return result

        restored_activations: dict[int, list[torch.Tensor]] = {
            l: [torch.zeros(0) for _ in restore_order] for l in self.layers
        }
        restored_num_tokens = [0] * len(restore_order)
        for sorted_pos, orig_pos in enumerate(restore_order):
            restored_num_tokens[orig_pos] = result.num_tokens[sorted_pos]
            for layer in self.layers:
                restored_activations[layer][orig_pos] = result.activations[layer][sorted_pos]

        return ActivationResult(
            activations=restored_activations,
            num_tokens=restored_num_tokens,
            timings=result.timings,
        )

    def _iter_flat_batch_payloads(
        self,
        inputs: list,
        *,
        batch_size: int | None,
        batch_token_budget: int | None,
        token_lens: list[int] | None,
        sampling_params: SamplingParams,
        include_timings: bool,
        use_staged_export: bool,
    ) -> Iterator[tuple[list[str], list[int], Any, dict[str, float]]]:
        """Yield raw flat payloads batch-by-batch in request order."""
        use_staged = (
            use_staged_export
            and self.staged_export
            and self.flat_output
            and self._local_worker is not None
            and hasattr(self._local_worker, "stage_captured_activations")
            and hasattr(self._local_worker, "pop_staged_activations")
        )

        pending_req_ids: list[str] | None = None
        pending_num_tokens: list[int] | None = None
        pending_timings: dict[str, float] | None = None
        batch_ranges = self._build_batch_ranges(
            total_inputs=len(inputs),
            batch_size=batch_size,
            batch_token_budget=batch_token_budget,
            token_lens=token_lens,
        )

        for start, end in batch_ranges:
            batch_inputs = inputs[start:end]
            req_ids, num_tokens, run_timings = self._run_prefill_batch(
                batch_inputs,
                sampling_params,
                include_timings,
            )

            if use_staged:
                t_stage0 = time.perf_counter()
                self._local_worker.stage_captured_activations()
                t_stage1 = time.perf_counter()
                if include_timings:
                    run_timings["stage_captured_activations_s"] = (
                        run_timings.get("stage_captured_activations_s", 0.0)
                        + (t_stage1 - t_stage0)
                    )

                if pending_req_ids is not None and pending_num_tokens is not None:
                    assert pending_timings is not None
                    t_pop0 = time.perf_counter()
                    payload = self._local_worker.pop_staged_activations()
                    t_pop1 = time.perf_counter()
                    if include_timings:
                        pending_timings["rpc_get_captured_activations_s"] = (
                            pending_timings.get("rpc_get_captured_activations_s", 0.0)
                            + (t_pop1 - t_pop0)
                        )
                    yield pending_req_ids, pending_num_tokens, payload, pending_timings

                pending_req_ids = req_ids
                pending_num_tokens = num_tokens
                pending_timings = run_timings if include_timings else {}
                continue

            t_fetch0 = time.perf_counter()
            payload = self._worker_call("get_captured_activations")[0]
            t_fetch1 = time.perf_counter()
            if include_timings:
                run_timings["rpc_get_captured_activations_s"] = (
                    run_timings.get("rpc_get_captured_activations_s", 0.0)
                    + (t_fetch1 - t_fetch0)
                )
                stats_results = self._worker_call("get_activation_capture_stats")
                if stats_results:
                    for k, v in stats_results[0].items():
                        run_timings[f"worker_{k}"] = (
                            run_timings.get(f"worker_{k}", 0.0) + float(v)
                        )
            yield req_ids, num_tokens, payload, run_timings if include_timings else {}

        if use_staged and pending_req_ids is not None and pending_num_tokens is not None:
            assert pending_timings is not None
            t_pop0 = time.perf_counter()
            payload = self._local_worker.pop_staged_activations()
            t_pop1 = time.perf_counter()
            if include_timings:
                pending_timings["rpc_get_captured_activations_s"] = (
                    pending_timings.get("rpc_get_captured_activations_s", 0.0)
                    + (t_pop1 - t_pop0)
                )
                stats_results = self._worker_call("get_activation_capture_stats")
                if stats_results:
                    for k, v in stats_results[0].items():
                        pending_timings[f"worker_{k}"] = (
                            pending_timings.get(f"worker_{k}", 0.0) + float(v)
                        )
            yield pending_req_ids, pending_num_tokens, payload, pending_timings

    def _payload_to_flat_batch(
        self,
        payload: Any,
        req_ids: list[str],
        num_tokens: list[int],
        timings: dict[str, float],
    ) -> FlatActivationResult:
        """Normalize worker payload into one chunked FlatActivationResult batch."""
        layers_payload: dict[int, dict[str, Any]] = {}

        if not isinstance(payload, dict) or payload.get("__flat__") is not True:
            acts = self._assemble_activations(payload, req_ids, num_tokens)
            for layer in self.layers:
                seqs = acts[layer]
                if seqs:
                    values = torch.cat(seqs, dim=0)
                    lengths = torch.tensor(
                        [int(t.shape[0]) for t in seqs], dtype=torch.int64
                    )
                else:
                    values = torch.empty((0, 0), dtype=torch.float32)
                    lengths = torch.zeros(len(req_ids), dtype=torch.int64)
                offsets = torch.empty(len(req_ids) + 1, dtype=torch.int64)
                offsets[0] = 0
                if len(req_ids) > 0:
                    offsets[1:] = torch.cumsum(lengths, dim=0)
                layers_payload[layer] = {
                    "__chunked__": True,
                    "values": [values],
                    "lengths": [lengths],
                    "offsets": offsets,
                }
            return FlatActivationResult(
                layers=layers_payload,
                num_tokens=list(num_tokens),
                req_ids=list(req_ids),
                timings=timings,
            )

        payload_req_ids = payload.get("req_ids", [])
        layer_payloads = payload.get("layers", {})
        if payload_req_ids == req_ids:
            req_positions = list(range(len(req_ids)))
        else:
            req_index = {rid: i for i, rid in enumerate(payload_req_ids)}
            req_positions = [req_index.get(rid, -1) for rid in req_ids]

        for layer in self.layers:
            layer_payload = layer_payloads.get(layer)
            if layer_payload is None:
                lengths = torch.zeros(len(req_ids), dtype=torch.int64)
                offsets = torch.empty(len(req_ids) + 1, dtype=torch.int64)
                offsets[0] = 0
                if len(req_ids) > 0:
                    offsets[1:] = torch.cumsum(lengths, dim=0)
                layers_payload[layer] = {
                    "__chunked__": True,
                    "values": [torch.empty((0, 0), dtype=torch.float32)],
                    "lengths": [lengths],
                    "offsets": offsets,
                }
                continue

            values = layer_payload["values"]
            offsets = layer_payload["offsets"]
            offsets_list = offsets.tolist()
            parts: list[torch.Tensor] = []
            lens_list: list[int] = []

            if req_positions == list(range(len(req_ids))):
                raw_lens = (offsets[1:].to(torch.int64) - offsets[:-1].to(torch.int64)).cpu()
                target_lens = torch.tensor(num_tokens, dtype=torch.int64)
                if raw_lens.numel() != target_lens.numel():
                    target_lens = target_lens[: raw_lens.numel()]
                if bool(torch.any(raw_lens > target_lens)):
                    for i, want in enumerate(num_tokens):
                        start = int(offsets_list[i])
                        end = int(offsets_list[i + 1])
                        keep_end = min(end, start + int(want))
                        keep_len = max(keep_end - start, 0)
                        lens_list.append(keep_len)
                        if keep_len > 0:
                            parts.append(values[start:keep_end])
                    lengths = torch.tensor(lens_list, dtype=torch.int64)
                else:
                    lengths = raw_lens
                    for i in range(len(req_ids)):
                        start = int(offsets_list[i])
                        end = int(offsets_list[i + 1])
                        if end > start:
                            parts.append(values[start:end])
            else:
                for i, pos in enumerate(req_positions):
                    if pos < 0:
                        lens_list.append(0)
                        continue
                    start = int(offsets_list[pos])
                    end = int(offsets_list[pos + 1])
                    keep_end = min(end, start + int(num_tokens[i]))
                    keep_len = max(keep_end - start, 0)
                    lens_list.append(keep_len)
                    if keep_len > 0:
                        parts.append(values[start:keep_end])
                lengths = torch.tensor(lens_list, dtype=torch.int64)

            if parts:
                batch_values = torch.cat(parts, dim=0)
            else:
                hidden = int(values.shape[1]) if values.ndim == 2 else 0
                batch_values = values.new_empty((0, hidden))
            if lengths.numel() != len(req_ids):
                if lengths.numel() < len(req_ids):
                    pad = torch.zeros(len(req_ids) - lengths.numel(), dtype=torch.int64)
                    lengths = torch.cat([lengths, pad], dim=0)
                else:
                    lengths = lengths[: len(req_ids)]
            offsets_out = torch.empty(len(req_ids) + 1, dtype=torch.int64)
            offsets_out[0] = 0
            if len(req_ids) > 0:
                offsets_out[1:] = torch.cumsum(lengths, dim=0)
            layers_payload[layer] = {
                "__chunked__": True,
                "values": [batch_values],
                "lengths": [lengths],
                "offsets": offsets_out,
            }

        return FlatActivationResult(
            layers=layers_payload,
            num_tokens=list(num_tokens),
            req_ids=list(req_ids),
            timings=timings,
        )

    def stream_flat(
        self,
        prompts: list[str] | None = None,
        token_ids: list[list[int]] | None = None,
        batch_size: int | None = None,
        batch_token_budget: int | None = None,
        pool_mode: str | None = None,
        include_timings: bool = False,
        sort_by_length: bool = True,
        preserve_input_order: bool = False,
        use_staged_export: bool = True,
    ) -> Iterator[FlatActivationResult]:
        """Yield flat activation payloads incrementally per processed batch."""
        if preserve_input_order:
            # Streaming reorder would require buffering all batches anyway.
            yield self.collect_flat(
                prompts=prompts,
                token_ids=token_ids,
                batch_size=batch_size,
                batch_token_budget=batch_token_budget,
                pool_mode=pool_mode,
                include_timings=include_timings,
                sort_by_length=sort_by_length,
                preserve_input_order=True,
                use_staged_export=use_staged_export,
            )
            return

        if self._closed:
            raise RuntimeError("ActivationEngine is closed")
        if prompts is None and token_ids is None:
            raise ValueError("Provide either prompts or token_ids")
        if prompts is not None and token_ids is not None:
            raise ValueError("Provide only one of prompts or token_ids")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_token_budget is not None and batch_token_budget <= 0:
            raise ValueError("batch_token_budget must be positive")
        if batch_token_budget is not None and token_ids is None:
            raise ValueError("batch_token_budget requires token_ids input")

        if token_ids is not None:
            inputs: list = [{"prompt_token_ids": ids} for ids in token_ids]
        else:
            inputs = prompts  # type: ignore[assignment]

        effective_pool_mode = self.pool_mode if pool_mode is None else pool_mode
        if effective_pool_mode not in (None, "mean", "last_token"):
            raise ValueError(
                f"Unsupported pool_mode={effective_pool_mode!r}. "
                "Expected one of: None, 'mean', 'last_token'."
            )
        if self._active_pool_mode != effective_pool_mode:
            self._register_capture(effective_pool_mode)

        ordered_token_lens: list[int] | None = None
        if token_ids is not None:
            ordered = self._ordered_indices(
                token_ids,
                batch_size,
                batch_token_budget,
                sort_by_length,
            )
            if ordered is not None:
                inputs = [inputs[i] for i in ordered]
                ordered_token_lens = [len(token_ids[i]) for i in ordered]
            else:
                ordered_token_lens = [len(ids) for ids in token_ids]

        sampling_params = self._make_sampling_params()
        for req_ids, num_tokens, payload, timings in self._iter_flat_batch_payloads(
            inputs,
            batch_size=batch_size,
            batch_token_budget=batch_token_budget,
            token_lens=ordered_token_lens,
            sampling_params=sampling_params,
            include_timings=include_timings,
            use_staged_export=use_staged_export,
        ):
            yield self._payload_to_flat_batch(
                payload,
                req_ids,
                num_tokens,
                timings if include_timings else {},
            )

    def collect_flat(
        self,
        prompts: list[str] | None = None,
        token_ids: list[list[int]] | None = None,
        batch_size: int | None = None,
        batch_token_budget: int | None = None,
        pool_mode: str | None = None,
        include_timings: bool = False,
        sort_by_length: bool = True,
        preserve_input_order: bool = False,
        use_staged_export: bool = True,
    ) -> FlatActivationResult:
        """Collect activations as flat layer payloads kept in memory."""
        if self._closed:
            raise RuntimeError("ActivationEngine is closed")
        if prompts is None and token_ids is None:
            raise ValueError("Provide either prompts or token_ids")
        if prompts is not None and token_ids is not None:
            raise ValueError("Provide only one of prompts or token_ids")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_token_budget is not None and batch_token_budget <= 0:
            raise ValueError("batch_token_budget must be positive")
        if batch_token_budget is not None and token_ids is None:
            raise ValueError("batch_token_budget requires token_ids input")

        if token_ids is not None:
            inputs: list = [{"prompt_token_ids": ids} for ids in token_ids]
        else:
            inputs = prompts  # type: ignore[assignment]

        effective_pool_mode = self.pool_mode if pool_mode is None else pool_mode
        if effective_pool_mode not in (None, "mean", "last_token"):
            raise ValueError(
                f"Unsupported pool_mode={effective_pool_mode!r}. "
                "Expected one of: None, 'mean', 'last_token'."
            )
        if self._active_pool_mode != effective_pool_mode:
            self._register_capture(effective_pool_mode)

        restore_order: list[int] | None = None
        ordered_token_lens: list[int] | None = None
        if token_ids is not None:
            ordered = self._ordered_indices(
                token_ids,
                batch_size,
                batch_token_budget,
                sort_by_length,
            )
            if ordered is not None:
                inputs = [inputs[i] for i in ordered]
                restore_order = ordered
                ordered_token_lens = [len(token_ids[i]) for i in ordered]
            else:
                ordered_token_lens = [len(ids) for ids in token_ids]

        sampling_params = self._make_sampling_params()
        result = self._collect_batches_flat(
            inputs,
            batch_size=batch_size,
            batch_token_budget=batch_token_budget,
            token_lens=ordered_token_lens,
            sampling_params=sampling_params,
            include_timings=include_timings,
            use_staged_export=use_staged_export,
        )

        if restore_order is None or not preserve_input_order:
            return result

        # restore_order maps sorted position -> original position.
        n = len(restore_order)
        orig_to_sorted = [0] * n
        for sorted_pos, orig_pos in enumerate(restore_order):
            orig_to_sorted[orig_pos] = sorted_pos

        reordered_layers: dict[int, dict[str, torch.Tensor]] = {}
        for layer in self.layers:
            payload = result.layers[layer]
            values = payload["values"]
            if isinstance(values, list):
                if values:
                    values = torch.cat(values, dim=0)
                else:
                    values = torch.empty((0, 0), dtype=torch.float32)
            offsets = payload["offsets"].tolist()
            lengths = [int(offsets[i + 1] - offsets[i]) for i in range(n)]
            reordered_lengths = [lengths[orig_to_sorted[i]] for i in range(n)]

            segs: list[torch.Tensor] = []
            for i in range(n):
                sidx = orig_to_sorted[i]
                start = int(offsets[sidx])
                end = int(offsets[sidx + 1])
                if end > start:
                    segs.append(values[start:end])
            if segs:
                reordered_values = torch.cat(segs, dim=0)
            else:
                hidden = int(values.shape[1]) if values.ndim == 2 else 0
                reordered_values = torch.empty((0, hidden), dtype=values.dtype)
            reordered_offsets = torch.empty(n + 1, dtype=torch.int64)
            reordered_offsets[0] = 0
            if n > 0:
                reordered_offsets[1:] = torch.cumsum(
                    torch.tensor(reordered_lengths, dtype=torch.int64),
                    dim=0,
                )
            reordered_layers[layer] = {
                "values": reordered_values,
                "offsets": reordered_offsets,
            }

        reordered_req_ids = [result.req_ids[orig_to_sorted[i]] for i in range(n)]
        reordered_num_tokens = [result.num_tokens[orig_to_sorted[i]] for i in range(n)]
        return FlatActivationResult(
            layers=reordered_layers,
            num_tokens=reordered_num_tokens,
            req_ids=reordered_req_ids,
            timings=result.timings,
        )

    def _collect_batches(
        self,
        inputs: list,
        batch_size: int | None,
        batch_token_budget: int | None,
        token_lens: list[int] | None,
        sampling_params: SamplingParams,
        include_timings: bool,
    ) -> ActivationResult:
        all_activations: dict[int, list[torch.Tensor]] = {l: [] for l in self.layers}
        all_num_tokens: list[int] = []
        timings: dict[str, float] = {}

        use_staged = (
            self.staged_export
            and self.flat_output
            and self._local_worker is not None
            and hasattr(self._local_worker, "stage_captured_activations")
            and hasattr(self._local_worker, "pop_staged_activations")
        )

        pending_req_ids: list[str] | None = None
        pending_num_tokens: list[int] | None = None
        batch_ranges = self._build_batch_ranges(
            total_inputs=len(inputs),
            batch_size=batch_size,
            batch_token_budget=batch_token_budget,
            token_lens=token_lens,
        )

        for start, end in batch_ranges:
            batch_inputs = inputs[start:end]
            req_ids, num_tokens, run_timings = self._run_prefill_batch(
                batch_inputs,
                sampling_params,
                include_timings,
            )
            if include_timings:
                for key, value in run_timings.items():
                    timings[key] = timings.get(key, 0.0) + value

            if use_staged:
                t_stage0 = time.perf_counter()
                self._local_worker.stage_captured_activations()
                t_stage1 = time.perf_counter()
                if include_timings:
                    timings["stage_captured_activations_s"] = (
                        timings.get("stage_captured_activations_s", 0.0)
                        + (t_stage1 - t_stage0)
                    )

                if pending_req_ids is not None and pending_num_tokens is not None:
                    t_pop0 = time.perf_counter()
                    payload = self._local_worker.pop_staged_activations()
                    t_pop1 = time.perf_counter()
                    if include_timings:
                        timings["rpc_get_captured_activations_s"] = (
                            timings.get("rpc_get_captured_activations_s", 0.0)
                            + (t_pop1 - t_pop0)
                        )
                    t_asm0 = time.perf_counter()
                    acts = self._assemble_activations(payload, pending_req_ids, pending_num_tokens)
                    t_asm1 = time.perf_counter()
                    if include_timings:
                        timings["python_assembly_s"] = (
                            timings.get("python_assembly_s", 0.0)
                            + (t_asm1 - t_asm0)
                        )
                    for layer in self.layers:
                        all_activations[layer].extend(acts[layer])
                    all_num_tokens.extend(pending_num_tokens)

                pending_req_ids = req_ids
                pending_num_tokens = num_tokens
                continue

            t_fetch0 = time.perf_counter()
            worker_results = self._worker_call("get_captured_activations")
            t_fetch1 = time.perf_counter()
            payload = worker_results[0]
            if include_timings:
                timings["rpc_get_captured_activations_s"] = (
                    timings.get("rpc_get_captured_activations_s", 0.0)
                    + (t_fetch1 - t_fetch0)
                )
                stats_results = self._worker_call("get_activation_capture_stats")
                if stats_results:
                    for k, v in stats_results[0].items():
                        timings[f"worker_{k}"] = timings.get(f"worker_{k}", 0.0) + float(v)

            t_asm0 = time.perf_counter()
            acts = self._assemble_activations(payload, req_ids, num_tokens)
            t_asm1 = time.perf_counter()
            if include_timings:
                timings["python_assembly_s"] = (
                    timings.get("python_assembly_s", 0.0) + (t_asm1 - t_asm0)
                )

            for layer in self.layers:
                all_activations[layer].extend(acts[layer])
            all_num_tokens.extend(num_tokens)

        if use_staged and pending_req_ids is not None and pending_num_tokens is not None:
            t_pop0 = time.perf_counter()
            payload = self._local_worker.pop_staged_activations()
            t_pop1 = time.perf_counter()
            if include_timings:
                timings["rpc_get_captured_activations_s"] = (
                    timings.get("rpc_get_captured_activations_s", 0.0)
                    + (t_pop1 - t_pop0)
                )
            t_asm0 = time.perf_counter()
            acts = self._assemble_activations(payload, pending_req_ids, pending_num_tokens)
            t_asm1 = time.perf_counter()
            if include_timings:
                timings["python_assembly_s"] = (
                    timings.get("python_assembly_s", 0.0) + (t_asm1 - t_asm0)
                )

            for layer in self.layers:
                all_activations[layer].extend(acts[layer])
            all_num_tokens.extend(pending_num_tokens)

            if include_timings:
                stats_results = self._worker_call("get_activation_capture_stats")
                if stats_results:
                    for k, v in stats_results[0].items():
                        timings[f"worker_{k}"] = timings.get(f"worker_{k}", 0.0) + float(v)

        return ActivationResult(
            activations=all_activations,
            num_tokens=all_num_tokens,
            timings=timings if include_timings else {},
        )

    def _collect_batches_flat(
        self,
        inputs: list,
        batch_size: int | None,
        batch_token_budget: int | None,
        token_lens: list[int] | None,
        sampling_params: SamplingParams,
        include_timings: bool,
        use_staged_export: bool,
    ) -> FlatActivationResult:
        timings: dict[str, float] = {}
        all_num_tokens: list[int] = []
        all_req_ids: list[str] = []
        value_parts: dict[int, list[torch.Tensor]] = {l: [] for l in self.layers}
        len_parts: dict[int, list[torch.Tensor]] = {l: [] for l in self.layers}

        use_staged = (
            use_staged_export
            and self.staged_export
            and self.flat_output
            and self._local_worker is not None
            and hasattr(self._local_worker, "stage_captured_activations")
            and hasattr(self._local_worker, "pop_staged_activations")
        )

        pending_req_ids: list[str] | None = None
        pending_num_tokens: list[int] | None = None
        batch_ranges = self._build_batch_ranges(
            total_inputs=len(inputs),
            batch_size=batch_size,
            batch_token_budget=batch_token_budget,
            token_lens=token_lens,
        )

        def merge_flat_payload(
            payload: Any,
            req_ids: list[str],
            num_tokens: list[int],
        ) -> None:
            if not isinstance(payload, dict) or payload.get("__flat__") is not True:
                acts = self._assemble_activations(payload, req_ids, num_tokens)
                for layer in self.layers:
                    seqs = acts[layer]
                    if seqs:
                        value_parts[layer].append(torch.cat(seqs, dim=0))
                        len_parts[layer].append(
                            torch.tensor([int(t.shape[0]) for t in seqs], dtype=torch.int64)
                        )
                    else:
                        len_parts[layer].append(torch.zeros(len(req_ids), dtype=torch.int64))
                return

            payload_req_ids = payload.get("req_ids", [])
            layer_payloads = payload.get("layers", {})
            if payload_req_ids == req_ids:
                req_positions = list(range(len(req_ids)))
            else:
                req_index = {rid: i for i, rid in enumerate(payload_req_ids)}
                req_positions = [req_index.get(rid, -1) for rid in req_ids]

            for layer in self.layers:
                layer_payload = layer_payloads.get(layer)
                if layer_payload is None:
                    len_parts[layer].append(torch.zeros(len(req_ids), dtype=torch.int64))
                    continue

                values = layer_payload["values"]
                offsets = layer_payload["offsets"]
                if req_positions == list(range(len(req_ids))):
                    raw_lens = offsets[1:].to(torch.int64) - offsets[:-1].to(torch.int64)
                    target_lens = torch.tensor(num_tokens, dtype=torch.int64)
                    if raw_lens.numel() != target_lens.numel():
                        target_lens = target_lens[: raw_lens.numel()]
                    # If any request captured an extra decode token, trim here so
                    # offsets remain aligned to prompt-token activations only.
                    if bool(torch.any(raw_lens > target_lens)):
                        offsets_list = offsets.tolist()
                        parts: list[torch.Tensor] = []
                        lens_list: list[int] = []
                        for i, want in enumerate(num_tokens):
                            start = int(offsets_list[i])
                            end = int(offsets_list[i + 1])
                            keep_end = min(end, start + int(want))
                            keep_len = max(keep_end - start, 0)
                            lens_list.append(keep_len)
                            if keep_len > 0:
                                parts.append(values[start:keep_end])
                        if parts:
                            value_parts[layer].append(torch.cat(parts, dim=0))
                        len_parts[layer].append(torch.tensor(lens_list, dtype=torch.int64))
                    else:
                        value_parts[layer].append(values)
                        len_parts[layer].append(raw_lens)
                    continue

                offsets_list = offsets.tolist()
                parts: list[torch.Tensor] = []
                lens_list: list[int] = []
                for i, pos in enumerate(req_positions):
                    if pos < 0:
                        lens_list.append(0)
                        continue
                    start = int(offsets_list[pos])
                    end = int(offsets_list[pos + 1])
                    keep_end = min(end, start + int(num_tokens[i]))
                    keep_len = max(keep_end - start, 0)
                    lens_list.append(keep_len)
                    if keep_len > 0:
                        parts.append(values[start:keep_end])
                if parts:
                    value_parts[layer].append(torch.cat(parts, dim=0))
                len_parts[layer].append(torch.tensor(lens_list, dtype=torch.int64))

        for start, end in batch_ranges:
            batch_inputs = inputs[start:end]
            req_ids, num_tokens, run_timings = self._run_prefill_batch(
                batch_inputs,
                sampling_params,
                include_timings,
            )
            if include_timings:
                for key, value in run_timings.items():
                    timings[key] = timings.get(key, 0.0) + value

            all_req_ids.extend(req_ids)
            all_num_tokens.extend(num_tokens)

            if use_staged:
                t_stage0 = time.perf_counter()
                self._local_worker.stage_captured_activations()
                t_stage1 = time.perf_counter()
                if include_timings:
                    timings["stage_captured_activations_s"] = (
                        timings.get("stage_captured_activations_s", 0.0)
                        + (t_stage1 - t_stage0)
                    )

                if pending_req_ids is not None and pending_num_tokens is not None:
                    t_pop0 = time.perf_counter()
                    payload = self._local_worker.pop_staged_activations()
                    t_pop1 = time.perf_counter()
                    if include_timings:
                        timings["rpc_get_captured_activations_s"] = (
                            timings.get("rpc_get_captured_activations_s", 0.0)
                            + (t_pop1 - t_pop0)
                        )
                    merge_flat_payload(payload, pending_req_ids, pending_num_tokens)

                pending_req_ids = req_ids
                pending_num_tokens = num_tokens
                continue

            t_fetch0 = time.perf_counter()
            payload = self._worker_call("get_captured_activations")[0]
            t_fetch1 = time.perf_counter()
            if include_timings:
                timings["rpc_get_captured_activations_s"] = (
                    timings.get("rpc_get_captured_activations_s", 0.0)
                    + (t_fetch1 - t_fetch0)
                )
            merge_flat_payload(payload, req_ids, num_tokens)

        if use_staged and pending_req_ids is not None and pending_num_tokens is not None:
            t_pop0 = time.perf_counter()
            payload = self._local_worker.pop_staged_activations()
            t_pop1 = time.perf_counter()
            if include_timings:
                timings["rpc_get_captured_activations_s"] = (
                    timings.get("rpc_get_captured_activations_s", 0.0)
                    + (t_pop1 - t_pop0)
                )
            merge_flat_payload(payload, pending_req_ids, pending_num_tokens)
            if include_timings:
                stats_results = self._worker_call("get_activation_capture_stats")
                if stats_results:
                    for k, v in stats_results[0].items():
                        timings[f"worker_{k}"] = timings.get(f"worker_{k}", 0.0) + float(v)

        layers_payload: dict[int, dict[str, Any]] = {}
        total_reqs = len(all_req_ids)
        for layer in self.layers:
            vals_list = value_parts[layer]
            lens_list = len_parts[layer]
            if lens_list:
                lengths = torch.cat(lens_list, dim=0)
            else:
                lengths = torch.zeros(total_reqs, dtype=torch.int64)

            if lengths.numel() != total_reqs:
                if lengths.numel() < total_reqs:
                    pad = torch.zeros(total_reqs - lengths.numel(), dtype=torch.int64)
                    lengths = torch.cat([lengths, pad], dim=0)
                else:
                    lengths = lengths[:total_reqs]

            offsets = torch.empty(total_reqs + 1, dtype=torch.int64)
            offsets[0] = 0
            if total_reqs > 0:
                offsets[1:] = torch.cumsum(lengths, dim=0)

            layers_payload[layer] = {
                "__chunked__": True,
                "values": vals_list,
                "lengths": lens_list,
                "offsets": offsets,
            }

        return FlatActivationResult(
            layers=layers_payload,
            num_tokens=all_num_tokens,
            req_ids=all_req_ids,
            timings=timings if include_timings else {},
        )

    def remove_hooks(self) -> None:
        """Remove all activation capture from the model."""
        self._worker_call("remove_activation_capture")

    def close(self) -> None:
        """Explicitly release hooks and engine resources."""
        if self._closed:
            return
        try:
            self.remove_hooks()
        except Exception:
            pass

        llm_engine = getattr(self.llm, "llm_engine", None)
        if llm_engine is not None:
            engine_core = getattr(llm_engine, "engine_core", None)
            if engine_core is not None and hasattr(engine_core, "shutdown"):
                try:
                    engine_core.shutdown()
                except Exception:
                    pass
            output_processor = getattr(llm_engine, "output_processor", None)
            if output_processor is not None and hasattr(output_processor, "close"):
                try:
                    output_processor.close()
                except Exception:
                    pass
            dp_group = getattr(llm_engine, "dp_group", None)
            if dp_group is not None:
                try:
                    from vllm.distributed import (
                        stateless_destroy_torch_distributed_process_group,
                    )

                    stateless_destroy_torch_distributed_process_group(dp_group)
                except Exception:
                    pass
            try:
                from vllm.distributed.parallel_state import (
                    destroy_distributed_environment,
                    destroy_model_parallel,
                )

                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception:
                pass

        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
