import argparse
import json
import statistics
import time
import types
from collections import Counter
from collections.abc import Callable

import torch


# Change only this function to benchmark a different forward workload.
def workload(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    return model(
        input_ids=input_ids,
        use_cache=False,
        logits_to_keep=1,
    ).logits


CONDITIONS = ("baseline", "optimized", "guards_disabled")
OPTIMIZATION_CONFIGS = {
    "recursive_tags": "use_recursive_dict_tags_for_guards",
    "versioned_lease": "use_versioned_guard_lease",
    "speculative": "speculative_guard_eval",
}


def summarize(samples: list[float]) -> dict[str, float | int]:
    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    margin = 1.984 * stdev / len(samples) ** 0.5
    return {
        "count": len(samples),
        "mean_ms": mean,
        "mean_ci95_low_ms": mean - margin,
        "mean_ci95_high_ms": mean + margin,
        "median_ms": statistics.median(samples),
        "stdev_ms": stdev,
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def summarize_reduction(
    baseline: list[float], guards_disabled: list[float]
) -> dict[str, float | int]:
    differences = [
        baseline_value - disabled_value
        for baseline_value, disabled_value in zip(
            baseline, guards_disabled, strict=True
        )
    ]
    result = summarize(differences)
    result["mean_percent"] = (
        100 * statistics.mean(differences) / statistics.mean(baseline)
    )
    return result


def clone_workload(name: str) -> Callable:
    return types.FunctionType(
        workload.__code__.replace(),
        workload.__globals__,
        name,
        workload.__defaults__,
        workload.__closure__,
    )


def compile_conditions(
    config_name: str,
) -> tuple[dict[str, Callable], dict[str, Callable]]:
    targets = {name: clone_workload(name) for name in CONDITIONS}
    compiled = {}
    for name in CONDITIONS:
        options = None
        if name == "guards_disabled":
            options = {"guard_filter_fn": torch.compiler.skip_all_guards_unsafe}
        with torch._dynamo.config.patch(**{config_name: name == "optimized"}):
            compiled[name] = torch.compile(
                targets[name], fullgraph=True, options=options
            )
    return targets, compiled


def measure_forward_latency(
    compiled: dict[str, Callable],
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    samples: int,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    total = {name: [] for name in CONDITIONS}
    cpu = {name: [] for name in CONDITIONS}
    for sample in range(samples):
        offset = sample % len(CONDITIONS)
        order = CONDITIONS[offset:] + CONDITIONS[:offset]
        for name in order:
            torch.cuda.synchronize(input_ids.device)
            start = time.perf_counter_ns()
            output = compiled[name](model, input_ids)
            submitted = time.perf_counter_ns()
            torch.cuda.synchronize(input_ids.device)
            finished = time.perf_counter_ns()
            cpu[name].append((submitted - start) / 1e6)
            total[name].append((finished - start) / 1e6)
            del output
    return total, cpu


def measure_guard_manager(
    guard_manager,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    samples: int,
) -> list[float]:
    frame_locals = {"model": model, "input_ids": input_ids}
    values = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        valid = guard_manager.root.check(frame_locals)
        values.append((time.perf_counter_ns() - start) / 1e6)
        if not valid:
            raise RuntimeError("guard manager unexpectedly rejected the input")
    return values


def count_leaf_guard_slots(manager) -> int:
    count = len(manager.get_leaf_guards())
    for child in manager.get_child_managers():
        count += count_leaf_guard_slots(child)
    if hasattr(manager, "get_epilogue_lambda_guards"):
        count += len(manager.get_epilogue_lambda_guards())
    return count


def count_accessor_types(manager) -> Counter[str]:
    counts = Counter(type(accessor).__name__ for accessor in manager.get_accessors())
    for child in manager.get_child_managers():
        counts.update(count_accessor_types(child))
    return counts


def summarize_root_subtrees(root) -> list[dict]:
    return [
        {
            "accessor": accessor.repr(),
            "leaf_guard_slots": count_leaf_guard_slots(child),
            "accessor_types": dict(count_accessor_types(child).most_common()),
        }
        for accessor, child in zip(
            root.get_accessors(), root.get_child_managers(), strict=True
        )
    ]


def measure_cpp_guard_timer(
    compiled: Callable,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    samples: int,
) -> dict[str, float | int] | None:
    eval_frame = torch._C._dynamo.eval_frame
    if not hasattr(eval_frame, "_debug_guard_eval_timing"):
        return None
    eval_frame._debug_guard_eval_timing(True)
    for _ in range(samples):
        compiled(model, input_ids)
    torch.cuda.synchronize(input_ids.device)
    timing = eval_frame._debug_guard_eval_timing(True)
    calls = timing["calls"]
    return {
        "count": calls,
        "mean_ms": timing["elapsed_ns"] / calls / 1e6 if calls else 0.0,
    }


def profile_runtime_ranges(
    compiled: Callable,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    samples: int,
) -> dict[str, dict[str, float | int] | None]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_stack=False,
    ) as profiler:
        for _ in range(samples):
            compiled(model, input_ids)
        torch.cuda.synchronize(input_ids.device)

    result = {}
    for name in (
        "TorchDynamo Cache Lookup",
        "TorchDynamo Frame Locals",
        "TorchDynamo Guard Evaluation",
        "TorchDynamo Guard Root Leaves",
        "TorchDynamo Guard Accessors",
        "TorchDynamo Guard Input Subtree",
        "TorchDynamo Guard Globals Subtree",
        "TorchDynamo Guard Model Subtree",
        "TorchDynamo Guard Epilogue",
        "Pregraph bytecode",
        "TorchDynamo Speculative Launch",
        "TorchDynamo Speculative Commit",
        "TorchDynamo Speculative Abort",
    ):
        values = [
            event.cpu_time_total / 1000
            for event in profiler.events()
            if event.name == name
        ]
        result[name] = summarize(values) if values else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--optimization", choices=tuple(OPTIMIZATION_CONFIGS), required=True
    )
    args = parser.parse_args()

    config_name = OPTIMIZATION_CONFIGS[args.optimization]
    if not hasattr(torch._dynamo.config, config_name):
        raise RuntimeError(
            f"this source revision does not provide torch._dynamo.config.{config_name}"
        )

    torchvision = torch.library.Library("torchvision", "DEF")
    torchvision.define(
        "nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor"
    )
    torchvision.define(
        "qnms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor"
    )
    from transformers import AutoModelForCausalLM

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval().requires_grad_(False).to(device)
    input_ids = torch.randint(
        model.config.vocab_size,
        (1, args.sequence_length),
        dtype=torch.int64,
        device=device,
    )

    torch._dynamo.reset()
    targets, compiled = compile_conditions(config_name)

    with torch.inference_mode():
        for name in CONDITIONS:
            with torch._dynamo.config.patch(
                **{config_name: name == "optimized"}
            ):
                compiled[name](model, input_ids)
        for sample in range(args.warmup):
            offset = sample % len(CONDITIONS)
            order = CONDITIONS[offset:] + CONDITIONS[:offset]
            for name in order:
                compiled[name](model, input_ids)
        torch.cuda.synchronize(device)

        guard_managers = {}
        for name in CONDITIONS:
            entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(
                targets[name]
            )
            if len(entries) != 1:
                raise RuntimeError(
                    f"expected one {name} cache entry, found {len(entries)}"
                )
            guard_managers[name] = entries[0].guard_manager

        total, cpu = measure_forward_latency(
            compiled, model, input_ids, args.samples
        )
        guards = {
            name: measure_guard_manager(
                guard_managers[name], model, input_ids, args.samples
            )
            for name in CONDITIONS
        }
        cpp_guard_timers = {
            name: measure_cpp_guard_timer(
                compiled[name], model, input_ids, args.samples
            )
            for name in CONDITIONS
        }
        runtime_ranges = {
            name: profile_runtime_ranges(
                compiled[name], model, input_ids, args.samples
            )
            for name in CONDITIONS
        }

    condition_results = {}
    for name in CONDITIONS:
        entry = torch._dynamo.eval_frame._debug_get_cache_entry_list(
            targets[name]
        )[0]
        descriptor = getattr(entry, "speculation_descriptor", None)
        condition_results[name] = {
            "total_forward_latency": summarize(total[name]),
            "cpu_submission_latency": summarize(cpu[name]),
            "direct_guard_manager_latency": summarize(guards[name]),
            "cpp_runtime_guard_latency": cpp_guard_timers[name],
            "runtime_ranges": runtime_ranges[name],
            "leaf_guard_slots": count_leaf_guard_slots(
                guard_managers[name].root
            ),
            "accessor_types": dict(
                count_accessor_types(guard_managers[name].root).most_common()
            ),
            "root_accessors": [
                accessor.repr()
                for accessor in guard_managers[name].root.get_accessors()
            ],
            "root_subtrees": summarize_root_subtrees(
                guard_managers[name].root
            ),
            "speculation_descriptor": (
                type(descriptor).__name__ if descriptor is not None else None
            ),
        }

    result = {
        "optimization": args.optimization,
        "model": args.model,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "input_shape": list(input_ids.shape),
        "samples_per_condition": args.samples,
        "torch_git_version": torch.version.git_version,
        "conditions": condition_results,
        "reductions": {
            "baseline_minus_optimized": {
                "total_forward_latency": summarize_reduction(
                    total["baseline"], total["optimized"]
                ),
                "cpu_submission_latency": summarize_reduction(
                    cpu["baseline"], cpu["optimized"]
                ),
                "direct_guard_manager_latency": summarize_reduction(
                    guards["baseline"], guards["optimized"]
                ),
            },
            "baseline_minus_guards_disabled": {
                "total_forward_latency": summarize_reduction(
                    total["baseline"], total["guards_disabled"]
                ),
                "cpu_submission_latency": summarize_reduction(
                    cpu["baseline"], cpu["guards_disabled"]
                ),
                "direct_guard_manager_latency": summarize_reduction(
                    guards["baseline"], guards["guards_disabled"]
                ),
            },
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
