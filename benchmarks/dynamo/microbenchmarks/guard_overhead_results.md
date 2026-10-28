# Qwen guard-overhead optimization results

## Setup

The workload is the `Qwen/Qwen3-0.6B` forward (596,049,920 parameters) in
BF16 with SDPA, batch 1, sequence length 128, and `logits_to_keep=1` on one
NVIDIA PG509-210 GPU. Each branch compiles three independent copies of the same
workload function: optimization disabled, optimization enabled, and all guards
unsafely disabled with `torch.compiler.skip_all_guards_unsafe`.

After 10 warmups, the benchmark rotates condition order and collects 100
CUDA-synchronized forward samples per condition. Savings and 95% confidence
intervals are computed from paired samples. Profiler ranges and standalone
guard microbenchmarks are collected separately. The profiler uses
`with_stack=False` to minimize measurement overhead. Absolute forward times
should not be compared across rows because each row required a separate
rebuild; the paired saving within each row is the primary result.

## Results

All times are milliseconds. `Forward saving` is baseline minus optimized.

| Optimization | Baseline forward | Optimized forward | Forward saving (95% CI) | Reduction | Cache lookup | Standalone guard microbench | Guardless ceiling | tlparse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Recursive dictionary tags | 24.242 | 23.760 | 0.482 (0.353 to 0.611) | 1.99% | 2.050 -> 1.607 | 0.518 -> 0.393 | 2.028 (8.37%) | [trace](https://manifold.edge.x2p.facebook.net/v0/read/tree/logs/guard-overhead-qwen-recursive-tags-20261002/index.html?bucketName=tlparse_reports&apiKey=tlparse_reports-key&withPayload=1&timeoutMsec=10000) |
| Versioned guard lease | 23.544 | 22.919 | 0.625 (0.501 to 0.749) | 2.65% | 1.931 -> 1.293 | 0.519 -> 0.320 | 1.927 (8.18%) | [trace](https://manifold.edge.x2p.facebook.net/v0/read/tree/logs/guard-overhead-qwen-versioned-lease-20261002/index.html?bucketName=tlparse_reports&apiKey=tlparse_reports-key&withPayload=1&timeoutMsec=10000) |
| Speculative guard evaluation | 24.395 | 24.293 | 0.103 (-0.136 to 0.342) | 0.42% | 2.098 -> 1.959 | 0.525 -> 0.498 | 2.152 (8.82%) | [trace](https://manifold.edge.x2p.facebook.net/v0/read/tree/logs/guard-overhead-qwen-speculative-20261002/index.html?bucketName=tlparse_reports&apiKey=tlparse_reports-key&withPayload=1&timeoutMsec=10000) |

Recursive tags and the lease produce statistically significant end-to-end
savings, but recover only part of the roughly 2 ms guardless ceiling. The lease
has the larger point estimate, although the 0.14 ms difference between the two
optimizations is not paired across builds and should not be treated as a proven
lease advantage.

The speculative implementation did not run speculatively on this graph: the
cache entry had no speculation descriptor and there were no speculative launch
or commit ranges. Qwen's value-dependent embedding read can asynchronously
assert for an invalid token, so the implementation's safety analysis rejects
the graph. Its confidence interval includes zero, as expected. This row
measures the safe automatic behavior, not the earlier manually forced
mechanism experiment.

Each tlparse was captured in a separate one-sample invocation so structured
tracing did not perturb the 100-sample timing runs.

Subsequent instrumentation placed a guard-evaluation range directly inside the
runtime cache lookup. It showed that guard evaluation accounts for about 99.8%
of cache lookup in this workload. The standalone `root.check()` measurements
above run repeatedly against a materialized dictionary and should be treated as
a hot-cache microbenchmark, not subtracted from runtime cache lookup. See
[the cache-lookup investigation](cache_lookup_investigation.md) for the runtime
breakdown.
