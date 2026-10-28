# Qwen cache-lookup investigation

This is an experimental checkpoint, not production-ready instrumentation. The
investigation runs on `origin/viable/strict` at
`4dff1b804743440f00de02cb1ddd7b41007a9381`. It uses the same Qwen3-0.6B
workload as the guard-overhead benchmark, plus temporary C++ profiler ranges
around cache lookup and the major guard-manager phases. The table is a
20-sample diagnostic run; the 100-sample cross-branch results remain the source
of truth for end-to-end gains.

All times are mean CPU milliseconds per invocation.

| Region | Baseline | Recursive tags enabled | Baseline share of guard evaluation |
| --- | ---: | ---: | ---: |
| Cache lookup | 1.917 | 1.225 | - |
| Guard evaluation | 1.914 | 1.223 | 100.0% |
| Root leaf guards | 0.028 | 0.026 | 1.4% |
| Recursive accessor traversal | 1.871 | 1.184 | 97.8% |
| Input subtree | 0.009 | 0.008 | 0.5% |
| Globals subtree | 0.227 | 0.184 | 11.9% |
| Model subtree | 1.624 | 0.983 | 84.9% |
| Epilogue guards | 0.001 | 0.001 | less than 0.1% |
| Cache machinery excluding guard evaluation | 0.003 | 0.002 | - |

The main finding is that `TorchDynamo Cache Lookup` is mostly a name for guard
evaluation in this workload. There is one cache entry, and roughly 99.8% of the
range is spent calling its root guard manager. Cache-list traversal, backend
comparison, result extraction, and LRU maintenance together account for only a
few microseconds. Constructing the frame-locals mapping is outside the lookup
range and takes about 0.002 ms.

Within guard evaluation, the model subtree is the clear target. It has 157 leaf
guard slots and recursively navigates the `nn.Module` state, including module
dictionaries and the 28 transformer layers. The globals subtree has 177 leaf
slots but is much faster, so raw guard count alone does not explain cost. The
recursive-tag optimization cuts the model subtree by 0.641 ms; this accounts
for about 93% of its 0.691 ms cache-lookup reduction.

The direct Python `root.check(frame_locals)` microbenchmark is not equivalent
to the actual runtime path: it measured 0.729 ms here while the runtime C++
guard call measured about 1.8 ms outside the profiler. Conclusions should use
the runtime ranges and C++ timer rather than treating the direct check as the
absolute lookup cost.

The next investigation step is to split the model subtree into configuration,
module hierarchy, and relational/alias checks, then attribute the expensive
module traversal to accessor types. This will distinguish whether the next safe
optimization should aggregate module mutation state, remove repeated lookup
work, or extend the recursive-tag/lease fast path.
