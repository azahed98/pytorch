"""
Compile-time profiler for Dynamo using Linux perf.

Captures both Python and C++ callstacks during compilation.

Usage:
    from torch._dynamo.compile_profiler import profile

    # Context manager
    with profile("compile.perf") as p:
        compiled = torch.compile(model)
        compiled(x)
    p.print_stats()

    # Decorator
    @profile("test.perf")
    def test_compile():
        ...

    # Function tracing
    from torch._dynamo.compile_profiler import trace_function
    tracer = trace_function("compile_check_fn")
    # ... run code ...
    tracer.report()
"""

from __future__ import annotations

import functools
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from types import FrameType


@dataclass
class ProfileSample:
    """A single sample from perf."""

    percent: float
    symbol: str
    library: str
    is_kernel: bool = False


@dataclass
class ProfileStats:
    """Parsed profiling results."""

    samples: list[ProfileSample] = field(default_factory=list)
    total_samples: int = 0
    duration_ms: float = 0.0
    perf_data_path: str = ""

    def print_stats(
        self,
        filter: Optional[str] = None,
        limit: int = 20,
        show_kernel: bool = True,
    ) -> None:
        """
        Print top functions by sample percentage.

        Args:
            filter: Only show symbols containing this string (case-insensitive)
            limit: Maximum number of entries to show
            show_kernel: Whether to include kernel symbols
        """
        print(f"\nProfile: {self.perf_data_path}")
        print(f"Duration: {self.duration_ms:.1f}ms, Samples: {self.total_samples}\n")
        print(f"{'%':>6}  {'Symbol':<50} Library")
        print("-" * 75)

        count = 0
        for sample in self.samples:
            if filter and filter.lower() not in sample.symbol.lower():
                continue
            if not show_kernel and sample.is_kernel:
                continue

            symbol_display = sample.symbol[:50]
            lib_display = sample.library[:20]
            print(f"{sample.percent:>5.1f}%  {symbol_display:<50} {lib_display}")
            count += 1
            if count >= limit:
                break

        if count == 0:
            print("  (no matching samples)")

    def top_by_category(self) -> dict[str, float]:
        """Group samples by category and return percentages."""
        categories: dict[str, float] = {
            "interpreter": 0.0,
            "gc": 0.0,
            "dict": 0.0,
            "guards": 0.0,
            "kernel": 0.0,
            "other": 0.0,
        }

        gc_symbols = {"deduce_unreachable", "visit_", "gc_collect", "untrack_"}
        guard_symbols = {"guard", "check_nopybind"}

        for sample in self.samples:
            sym_lower = sample.symbol.lower()

            if sample.is_kernel:
                categories["kernel"] += sample.percent
            elif "pyeval" in sym_lower or "evalframe" in sym_lower:
                categories["interpreter"] += sample.percent
            elif any(gc in sym_lower for gc in gc_symbols):
                categories["gc"] += sample.percent
            elif "dict" in sym_lower:
                categories["dict"] += sample.percent
            elif any(g in sym_lower for g in guard_symbols):
                categories["guards"] += sample.percent
            else:
                categories["other"] += sample.percent

        return categories

    def print_summary(self) -> None:
        """Print a high-level summary by category."""
        categories = self.top_by_category()
        print(f"\nProfile Summary: {self.perf_data_path}")
        print("-" * 40)
        for cat, pct in sorted(categories.items(), key=lambda x: -x[1]):
            if pct > 0.1:
                bar = "█" * int(pct / 2)
                print(f"{cat:<12} {pct:>5.1f}% {bar}")

    def export_flamegraph(self, output_path: str) -> bool:
        """
        Generate a flame graph SVG.

        Requires stackcollapse-perf.pl and flamegraph.pl in PATH.
        Get them from: https://github.com/brendangregg/FlameGraph

        Args:
            output_path: Path for output SVG file

        Returns:
            True if successful, False otherwise
        """
        if not self.perf_data_path or not os.path.exists(self.perf_data_path):
            print(f"Error: perf data file not found: {self.perf_data_path}")
            return False

        # Check for flamegraph tools
        stackcollapse = shutil.which("stackcollapse-perf.pl")
        flamegraph = shutil.which("flamegraph.pl")

        if not stackcollapse or not flamegraph:
            print("Error: FlameGraph tools not found in PATH")
            print("Install from: https://github.com/brendangregg/FlameGraph")
            print("\nManual command:")
            print(
                f"  perf script -i {self.perf_data_path} | "
                f"stackcollapse-perf.pl | flamegraph.pl > {output_path}"
            )
            return False

        try:
            # perf script | stackcollapse-perf.pl | flamegraph.pl > output.svg
            perf_script = subprocess.Popen(
                ["perf", "script", "-i", self.perf_data_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            collapse = subprocess.Popen(
                [stackcollapse],
                stdin=perf_script.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            with open(output_path, "w") as f:
                flamegraph_proc = subprocess.run(
                    [flamegraph],
                    stdin=collapse.stdout,
                    stdout=f,
                    stderr=subprocess.DEVNULL,
                )

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Flame graph written to: {output_path}")
                return True
            else:
                print("Error: Failed to generate flame graph")
                return False

        except Exception as e:
            print(f"Error generating flame graph: {e}")
            return False


def _perf_available() -> bool:
    """Check if perf is available."""
    return shutil.which("perf") is not None


def _trampoline_available() -> bool:
    """Check if Python perf trampoline is available (Python 3.12+)."""
    return hasattr(sys, "activate_stack_trampoline")


class ProfileSession:
    """
    Manages a profiling session using Linux perf.

    Captures both Python and C++ callstacks by:
    1. Enabling Python's perf trampoline (makes Python functions visible)
    2. Running perf record attached to the current process
    3. Parsing perf report output

    Usage:
        with ProfileSession("output.perf") as session:
            # code to profile
            ...
        session.print_stats()
    """

    def __init__(
        self,
        output: str = "profile.perf",
        frequency: int = 999,
        call_graph: str = "fp",
    ):
        """
        Initialize a profiling session.

        Args:
            output: Path for perf.data output file
            frequency: Sampling frequency in Hz (default 999)
            call_graph: Call graph mode - "fp" (fast, default) or "dwarf" (accurate)
        """
        self.output = output
        self.frequency = frequency
        self.call_graph = call_graph
        self._perf_proc: Optional[subprocess.Popen[bytes]] = None
        self._stats: Optional[ProfileStats] = None
        self._start_time: float = 0.0
        self._trampoline_was_active: bool = False

    def __enter__(self) -> "ProfileSession":
        if not _perf_available():
            import warnings

            warnings.warn(
                "perf not available, profiling disabled. "
                "Install linux-perf or perf-tools.",
                stacklevel=2,
            )
            return self

        # Check if trampoline already active
        if _trampoline_available():
            self._trampoline_was_active = sys.is_stack_trampoline_active()
            if not self._trampoline_was_active:
                sys.activate_stack_trampoline("perf")

        # Build perf command
        cmd = [
            "perf",
            "record",
            "-o",
            self.output,
            "-F",
            str(self.frequency),
            "-p",
            str(os.getpid()),
        ]

        if self.call_graph == "dwarf":
            cmd.extend(["--call-graph", "dwarf,16384"])
        else:
            cmd.append("-g")

        # Start perf record
        self._perf_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Give perf time to attach
        time.sleep(0.05)
        self._start_time = time.perf_counter()

        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        duration_ms = (time.perf_counter() - self._start_time) * 1000

        # Stop perf - send SIGINT and wait for clean shutdown
        if self._perf_proc:
            # Give perf time to flush any pending samples
            time.sleep(0.1)

            self._perf_proc.send_signal(signal.SIGINT)
            try:
                self._perf_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Try SIGTERM before SIGKILL
                self._perf_proc.terminate()
                try:
                    self._perf_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._perf_proc.kill()
                    self._perf_proc.wait()

            # Small delay to ensure file is written
            time.sleep(0.1)

        # Deactivate trampoline if we activated it
        if _trampoline_available() and not self._trampoline_was_active:
            if sys.is_stack_trampoline_active():
                sys.deactivate_stack_trampoline()

        # Parse results
        if os.path.exists(self.output):
            self._stats = self._parse_results(duration_ms)

    def _parse_results(self, duration_ms: float) -> ProfileStats:
        """Parse perf report output into ProfileStats."""
        result = subprocess.run(
            ["perf", "report", "--stdio", "-i", self.output],
            capture_output=True,
            text=True,
        )

        samples = []
        for line in result.stdout.split("\n"):
            # Strip leading/trailing whitespace
            line = line.strip()

            if "%" not in line:
                continue
            if "[.]" not in line and "[k]" not in line:
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            try:
                pct = float(parts[0].rstrip("%"))
                # Symbol is the last part
                symbol = parts[-1]
                # Library is typically parts[2] (after command name)
                lib = parts[2] if len(parts) > 3 else "unknown"
                is_kernel = "[k]" in line

                samples.append(
                    ProfileSample(
                        percent=pct,
                        symbol=symbol,
                        library=lib,
                        is_kernel=is_kernel,
                    )
                )
            except (ValueError, IndexError):
                continue

        return ProfileStats(
            samples=samples,
            total_samples=len(samples),
            duration_ms=duration_ms,
            perf_data_path=self.output,
        )

    def print_stats(
        self,
        filter: Optional[str] = None,
        limit: int = 20,
    ) -> None:
        """Print profiling results."""
        if self._stats:
            self._stats.print_stats(filter=filter, limit=limit)
        else:
            print("No profiling data available")

    def print_summary(self) -> None:
        """Print high-level summary by category."""
        if self._stats:
            self._stats.print_summary()
        else:
            print("No profiling data available")

    def export_flamegraph(self, output_path: str) -> bool:
        """Generate flame graph SVG."""
        if self._stats:
            return self._stats.export_flamegraph(output_path)
        print("No profiling data available")
        return False

    @property
    def stats(self) -> Optional[ProfileStats]:
        """Get the parsed profiling statistics."""
        return self._stats


class _ProfileDecorator:
    """
    Wrapper that works as both context manager and decorator.

    This allows:
        @profile("file.perf")
        def func(): ...

        with profile("file.perf") as p: ...
    """

    def __init__(
        self,
        output: str = "profile.perf",
        frequency: int = 999,
        call_graph: str = "dwarf",
    ):
        self.output = output
        self.frequency = frequency
        self.call_graph = call_graph
        self._session: Optional[ProfileSession] = None

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Use as decorator: @profile("file.perf")"""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with ProfileSession(
                output=self.output,
                frequency=self.frequency,
                call_graph=self.call_graph,
            ) as session:
                result = func(*args, **kwargs)
            session.print_stats()
            return result

        return wrapper

    def __enter__(self) -> ProfileSession:
        """Use as context manager: with profile("file.perf") as p:"""
        self._session = ProfileSession(
            output=self.output,
            frequency=self.frequency,
            call_graph=self.call_graph,
        )
        return self._session.__enter__()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        if self._session:
            self._session.__exit__(exc_type, exc_val, exc_tb)


# Type for the profile() function which can be used as decorator or context manager
_ProfileTarget = Union[str, Callable[..., Any]]


def profile(
    output: _ProfileTarget = "profile.perf",
    frequency: int = 999,
    call_graph: str = "dwarf",
) -> Union[_ProfileDecorator, Callable[..., Any]]:
    """
    Profile a code block or function, capturing Python and C++ callstacks.

    Can be used as a context manager or decorator.

    As context manager:
        with profile("compile.perf") as p:
            compiled = torch.compile(model)
            compiled(x)
        p.print_stats()
        p.print_stats("guard")  # Filter by keyword

    As decorator:
        @profile("test.perf")
        def test_my_function():
            ...

        # Or without arguments:
        @profile
        def test_my_function():
            ...

    Args:
        output: Path for perf.data file, or function if used as @profile
        frequency: Sampling frequency in Hz
        call_graph: "dwarf" (accurate C++ stacks) or "fp" (faster)

    Returns:
        Context manager/decorator, or decorated function
    """
    # Handle @profile without parentheses
    if callable(output):
        func = output

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with ProfileSession() as session:
                result = func(*args, **kwargs)
            session.print_stats()
            return result

        return wrapper

    # Return wrapper that works as both context manager and decorator
    return _ProfileDecorator(output=output, frequency=frequency, call_graph=call_graph)


@dataclass
class FunctionCall:
    """Record of a single function call."""

    start_ns: int
    end_ns: int
    duration_ns: int


@dataclass
class FunctionStats:
    """Statistics for a traced function."""

    calls: list[FunctionCall] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.calls)

    @property
    def total_ms(self) -> float:
        return sum(c.duration_ns for c in self.calls) / 1e6

    @property
    def avg_ms(self) -> float:
        return self.total_ms / max(1, self.count)

    @property
    def min_ms(self) -> float:
        if not self.calls:
            return 0.0
        return min(c.duration_ns for c in self.calls) / 1e6

    @property
    def max_ms(self) -> float:
        if not self.calls:
            return 0.0
        return max(c.duration_ns for c in self.calls) / 1e6


class FunctionTracer:
    """
    Trace specific function calls with precise timing.

    Uses sys.monitoring (Python 3.12+) for low-overhead tracing.

    Usage:
        tracer = FunctionTracer()
        tracer.trace("compile_check_fn")

        # ... run code that calls compile_check_fn ...

        tracer.report()
        tracer.stop()
    """

    _TOOL_ID = 3  # Use tool ID 3 (custom tool)

    def __init__(self) -> None:
        self._targets: dict[str, FunctionStats] = {}
        self._active_calls: dict[str, list[int]] = {}  # Stack of start times
        self._monitoring_active = False

    def trace(self, func_name: str) -> "FunctionTracer":
        """
        Start tracing calls to a function.

        Args:
            func_name: Function name or partial path to match
                       e.g., "compile_check_fn" or "guards.py:compile_check_fn"

        Returns:
            self for chaining
        """
        if not hasattr(sys, "monitoring"):
            import warnings

            warnings.warn(
                "sys.monitoring not available (requires Python 3.12+). "
                "Function tracing disabled.",
                stacklevel=2,
            )
            return self

        self._targets[func_name] = FunctionStats()
        self._active_calls[func_name] = []

        if not self._monitoring_active:
            self._start_monitoring()

        return self

    def _start_monitoring(self) -> None:
        """Set up sys.monitoring callbacks."""
        if not hasattr(sys, "monitoring"):
            return

        monitoring = sys.monitoring

        # Register our tool
        monitoring.use_tool_id(self._TOOL_ID, "dynamo_compile_profiler")

        # Set events we care about
        monitoring.set_events(
            self._TOOL_ID,
            monitoring.events.PY_START | monitoring.events.PY_RETURN,
        )

        # Register callbacks
        monitoring.register_callback(
            self._TOOL_ID,
            monitoring.events.PY_START,
            self._on_call,
        )
        monitoring.register_callback(
            self._TOOL_ID,
            monitoring.events.PY_RETURN,
            self._on_return,
        )

        self._monitoring_active = True

    def _on_call(self, code: Any, instruction_offset: int) -> None:
        """Called on function entry."""
        func_id = f"{code.co_filename}:{code.co_name}"

        for target in self._targets:
            if target in func_id or target == code.co_name:
                self._active_calls[target].append(time.perf_counter_ns())
                break

    def _on_return(self, code: Any, instruction_offset: int, retval: Any) -> None:
        """Called on function return."""
        end_ns = time.perf_counter_ns()
        func_id = f"{code.co_filename}:{code.co_name}"

        for target in self._targets:
            if target in func_id or target == code.co_name:
                if self._active_calls[target]:
                    start_ns = self._active_calls[target].pop()
                    duration_ns = end_ns - start_ns
                    self._targets[target].calls.append(
                        FunctionCall(
                            start_ns=start_ns,
                            end_ns=end_ns,
                            duration_ns=duration_ns,
                        )
                    )
                break

    def stop(self) -> None:
        """Stop monitoring."""
        if not self._monitoring_active:
            return

        if hasattr(sys, "monitoring"):
            sys.monitoring.set_events(self._TOOL_ID, 0)
            self._monitoring_active = False

    def report(self) -> None:
        """Print timing report for all traced functions."""
        print("\nFunction Trace Report")
        print("=" * 50)

        for name, stats in self._targets.items():
            if not stats.calls:
                print(f"\n{name}: no calls recorded")
                continue

            print(f"\n{name}:")
            print(f"  Calls:  {stats.count}")
            print(f"  Total:  {stats.total_ms:.2f}ms")
            print(f"  Avg:    {stats.avg_ms:.2f}ms")
            print(f"  Min:    {stats.min_ms:.2f}ms")
            print(f"  Max:    {stats.max_ms:.2f}ms")

    def get_stats(self, func_name: str) -> Optional[FunctionStats]:
        """Get statistics for a specific function."""
        return self._targets.get(func_name)

    def clear(self) -> None:
        """Clear all recorded data."""
        for stats in self._targets.values():
            stats.calls.clear()
        for calls in self._active_calls.values():
            calls.clear()


# Global tracer instance for convenience
_global_tracer: Optional[FunctionTracer] = None


def trace_function(func_name: str) -> FunctionTracer:
    """
    Trace a specific function by name.

    Convenience function that uses a global FunctionTracer.

    Usage:
        tracer = trace_function("compile_check_fn")
        # ... run code ...
        tracer.report()

    Args:
        func_name: Function name to trace (partial match supported)

    Returns:
        FunctionTracer instance
    """
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = FunctionTracer()
    _global_tracer.trace(func_name)
    return _global_tracer


@contextmanager
def profile_region(name: str = "region"):
    """
    Simple context manager to time a region of code.

    Unlike profile(), this doesn't use perf - just wall clock timing.
    Useful for quick measurements.

    Usage:
        with profile_region("guard_creation") as t:
            # ... code ...
        print(f"Took {t.duration_ms:.2f}ms")
    """

    @dataclass
    class RegionTimer:
        name: str
        start_ns: int = 0
        end_ns: int = 0

        @property
        def duration_ms(self) -> float:
            return (self.end_ns - self.start_ns) / 1e6

    timer = RegionTimer(name=name)
    timer.start_ns = time.perf_counter_ns()
    try:
        yield timer
    finally:
        timer.end_ns = time.perf_counter_ns()
