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

    # Filter by category (interpreter, gc, dict, attr, guards, tensor, dynamo)
    p.print_stats(category="gc")

    # Filter by caller function (only samples under a specific frame)
    p.print_stats(caller="compile_check_fn")

    # Combine filters
    p.print_stats(category="dict", caller="compile_check_fn")

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


# Categories for filtering and grouping
CATEGORY_PATTERNS: dict[str, list[str]] = {
    "interpreter": ["pyeval", "evalframe", "_PyFunction_Vectorcall", "_PyEval_"],
    "gc": ["deduce_unreachable", "visit_", "gc_collect", "untrack_", "_PyGC_"],
    "dict": ["dict_lookup", "dict_get", "dict_set", "_PyDict_", "lookdict"],
    "attr": ["getattr", "setattr", "_PyObject_GenericGetAttr", "PyObject_GetAttr"],
    "guards": ["guard", "check_nopybind", "GuardManager", "CheckFunction"],
    "tensor": ["TensorImpl", "aten::", "ATen", "c10::"],
    "dynamo": ["_dynamo", "torch/dynamo", "torch._dynamo"],
}


@dataclass
class ProfileSample:
    """A single sample from perf."""

    percent: float
    symbol: str
    library: str
    is_kernel: bool = False
    stack: list[str] = field(default_factory=list)  # Full call stack if available

    def get_category(self) -> str:
        """Determine the category of this sample."""
        sym_lower = self.symbol.lower()
        lib_lower = self.library.lower()
        combined = f"{sym_lower} {lib_lower}"

        if self.is_kernel:
            return "kernel"

        for category, patterns in CATEGORY_PATTERNS.items():
            if any(p.lower() in combined for p in patterns):
                return category

        return "other"


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
        category: Optional[str] = None,
        limit: int = 20,
        show_kernel: bool = True,
    ) -> None:
        """
        Print top functions by sample percentage.

        Args:
            filter: Only show symbols containing this string (case-insensitive)
            category: Only show samples in this category
                      (interpreter, gc, dict, attr, guards, tensor, dynamo, kernel, other)
            limit: Maximum number of entries to show
            show_kernel: Whether to include kernel symbols
        """
        print(f"\nProfile: {self.perf_data_path}")
        print(f"Duration: {self.duration_ms:.1f}ms, Samples: {self.total_samples}")

        if category:
            print(f"Filtering by category: {category}")
        if filter:
            print(f"Filtering by: '{filter}'")
        print()

        print(f"{'%':>6}  {'Cat':<6}  {'Symbol':<44} Library")
        print("-" * 80)

        count = 0
        filtered_pct = 0.0
        for sample in self.samples:
            sample_cat = sample.get_category()

            if filter and filter.lower() not in sample.symbol.lower():
                continue
            if category and sample_cat != category:
                continue
            if not show_kernel and sample.is_kernel:
                continue

            filtered_pct += sample.percent
            symbol_display = sample.symbol[:44]
            lib_display = sample.library[:16]
            cat_display = sample_cat[:6]
            print(f"{sample.percent:>5.1f}%  {cat_display:<6}  {symbol_display:<44} {lib_display}")
            count += 1
            if count >= limit:
                remaining = sum(
                    s.percent
                    for s in self.samples[count:]
                    if (not category or s.get_category() == category)
                    and (not filter or filter.lower() in s.symbol.lower())
                )
                if remaining > 0:
                    print(f"  ... and {remaining:.1f}% more")
                break

        if count == 0:
            print("  (no matching samples)")
        elif category or filter:
            print(f"\nTotal in filter: {filtered_pct:.1f}%")

    def top_by_category(self) -> dict[str, float]:
        """Group samples by category and return percentages."""
        categories: dict[str, float] = {}
        for cat in list(CATEGORY_PATTERNS.keys()) + ["kernel", "other"]:
            categories[cat] = 0.0

        for sample in self.samples:
            cat = sample.get_category()
            categories[cat] = categories.get(cat, 0.0) + sample.percent

        return categories

    def filter_by_caller(self, caller: str) -> "ProfileStats":
        """
        Return a new ProfileStats with only samples that occurred under a specific caller.

        This requires that samples have stack information (call_graph mode was used).

        Args:
            caller: Function name or pattern to match in the call stack

        Returns:
            New ProfileStats with filtered samples, percentages recalculated
        """
        filtered = [s for s in self.samples if self._sample_under_caller(s, caller)]

        if not filtered:
            return ProfileStats(
                samples=[],
                total_samples=0,
                duration_ms=self.duration_ms,
                perf_data_path=self.perf_data_path,
            )

        # Recalculate percentages relative to filtered set
        total_pct = sum(s.percent for s in filtered)
        if total_pct > 0:
            scale = 100.0 / total_pct
            new_samples = [
                ProfileSample(
                    percent=s.percent * scale,
                    symbol=s.symbol,
                    library=s.library,
                    is_kernel=s.is_kernel,
                    stack=s.stack,
                )
                for s in filtered
            ]
        else:
            new_samples = filtered

        return ProfileStats(
            samples=new_samples,
            total_samples=len(new_samples),
            duration_ms=self.duration_ms,
            perf_data_path=self.perf_data_path + f" [caller={caller}]",
        )

    def _sample_under_caller(self, sample: ProfileSample, caller: str) -> bool:
        """Check if sample occurred while caller was on the stack."""
        caller_lower = caller.lower()

        # Check the stack if available
        if sample.stack:
            return any(caller_lower in frame.lower() for frame in sample.stack)

        # Fallback: check if symbol itself matches
        return caller_lower in sample.symbol.lower()

    def analyze_by_caller(self, caller: str) -> dict[str, Any]:
        """
        Analyze samples by caller directly from the perf data.

        This parses perf script output and counts how many samples
        occurred while the caller was on the stack.

        Args:
            caller: Function name pattern to match

        Returns:
            Dict with 'total_samples', 'under_caller', 'top_symbols' (counter)
        """
        if not self.perf_data_path or not os.path.exists(
            self.perf_data_path.split()[0]
        ):
            return {"total_samples": 0, "under_caller": 0}

        perf_file = self.perf_data_path.split()[0]  # Handle "[caller=x]" suffix
        try:
            # Use -F to get better symbol output
            result = subprocess.run(
                ["perf", "script", "-i", perf_file, "-F", "comm,ip,sym,dso"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return {"total_samples": 0, "under_caller": 0}

        caller_lower = caller.lower()
        total_samples = 0
        under_caller = 0
        symbol_counts: dict[str, int] = {}
        current_stack: list[str] = []
        in_sample = False

        for line in result.stdout.split("\n"):
            stripped = line.rstrip()

            if stripped and not stripped.startswith(("\t", " ")):
                # New sample - process previous
                if current_stack:
                    total_samples += 1
                    if any(caller_lower in frame.lower() for frame in current_stack):
                        under_caller += 1
                        # Count top symbol (extract full symbol name)
                        if current_stack:
                            top = self._extract_symbol_from_frame(current_stack[0])
                            symbol_counts[top] = symbol_counts.get(top, 0) + 1
                current_stack = []
                in_sample = True
            elif in_sample and stripped.startswith(("\t", " ")):
                current_stack.append(stripped)
            elif not stripped:
                in_sample = False

        # Handle last sample
        if current_stack:
            total_samples += 1
            if any(caller_lower in frame.lower() for frame in current_stack):
                under_caller += 1

        return {
            "total_samples": total_samples,
            "under_caller": under_caller,
            "top_symbols": symbol_counts,
        }

    @staticmethod
    def _extract_symbol_from_frame(frame: str) -> str:
        """
        Extract the symbol name from a perf script frame line.

        Frame format: "    7fa9b5ac21a _PyEval_EvalFrameDefault (/path/to/python)"
        """
        parts = frame.strip().split()
        if len(parts) < 2:
            return frame.strip()

        # parts[0] is the address, parts[1] is the symbol
        symbol = parts[1]

        # If symbol is [unknown], try to use the DSO name
        if symbol == "[unknown]" and len(parts) >= 3:
            # Get DSO from (path/to/lib)
            dso = " ".join(parts[2:])
            if "(" in dso:
                dso = dso.split("(")[-1].rstrip(")")
                # Extract just the filename
                dso = dso.split("/")[-1]
                return f"[{dso}]"

        return symbol

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
        # First get the flat report for percentages
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

        # Try to get stack information for caller filtering
        if self.call_graph in ("dwarf", "fp"):
            self._enrich_with_stacks(samples)

        return ProfileStats(
            samples=samples,
            total_samples=len(samples),
            duration_ms=duration_ms,
            perf_data_path=self.output,
        )

    def _enrich_with_stacks(self, samples: list[ProfileSample]) -> None:
        """
        Enrich samples with stack information from perf script.

        This allows filtering by caller function.
        """
        try:
            result = subprocess.run(
                ["perf", "script", "-i", self.output],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return

        # Parse all samples from perf script and build symbol -> stacks map
        # perf script format:
        #   python 12345 [000] 12345.678901: cycles:
        #       7fff12345678 symbol1+0x10 (library1)
        #       7fff12345679 symbol2+0x20 (library2)
        #       ...
        symbol_stacks: dict[str, list[list[str]]] = {}
        current_stack: list[str] = []
        in_sample = False

        for line in result.stdout.split("\n"):
            stripped = line.rstrip()

            # Detect sample header (not indented, contains process info)
            if stripped and not stripped.startswith(("\t", " ")):
                # Save previous sample
                if current_stack:
                    self._save_stack(symbol_stacks, current_stack)
                current_stack = []
                in_sample = True
            elif in_sample and (stripped.startswith("\t") or stripped.startswith(" ")):
                # Stack frame line
                # Format: "\t7fff12345678 symbol+0x10 (/path/to/lib)"
                # Or:    "            7fff12345678 symbol+0x10 (/path/to/lib)"
                parts = stripped.split()
                if len(parts) >= 2:
                    # Extract full frame info including library path
                    frame_info = " ".join(parts[1:])  # Everything after address
                    current_stack.append(frame_info)
            elif not stripped:
                # Empty line - end of sample
                if current_stack:
                    self._save_stack(symbol_stacks, current_stack)
                current_stack = []
                in_sample = False

        # Handle last sample
        if current_stack:
            self._save_stack(symbol_stacks, current_stack)

        # Associate stacks with samples (match by symbol name)
        for sample in samples:
            base_symbol = sample.symbol.split("+")[0]
            if base_symbol in symbol_stacks and symbol_stacks[base_symbol]:
                sample.stack = symbol_stacks[base_symbol][0]

    def _save_stack(
        self, symbol_stacks: dict[str, list[list[str]]], stack: list[str]
    ) -> None:
        """Save a stack trace, keyed by the top (sampled) symbol."""
        if not stack:
            return
        # Extract symbol from first frame (format: "symbol+0x10 (/path/to/lib)")
        top_frame = stack[0]
        # Symbol is before '+' or space
        if "+" in top_frame:
            top_symbol = top_frame.split("+")[0]
        else:
            top_symbol = top_frame.split()[0] if " " in top_frame else top_frame

        if top_symbol not in symbol_stacks:
            symbol_stacks[top_symbol] = []
        symbol_stacks[top_symbol].append(stack.copy())

    def print_stats(
        self,
        filter: Optional[str] = None,
        category: Optional[str] = None,
        caller: Optional[str] = None,
        limit: int = 20,
    ) -> None:
        """
        Print profiling results.

        Args:
            filter: Only show symbols containing this string
            category: Only show samples in this category
                      (interpreter, gc, dict, attr, guards, tensor, dynamo, kernel, other)
            caller: Only show samples that occurred under this function
            limit: Maximum entries to show
        """
        if not self._stats:
            print("No profiling data available")
            return

        stats = self._stats
        if caller:
            # Use analyze_by_caller for more reliable results
            analysis = stats.analyze_by_caller(caller)
            under_caller = analysis.get("under_caller", 0)
            total = analysis.get("total_samples", 0)

            if under_caller == 0:
                print(f"No samples found under caller '{caller}'")
                print("Note: caller filtering requires call graph mode (dwarf/fp)")
                return

            # Get top symbols and create synthetic stats for display
            top_symbols = analysis.get("top_symbols", {})
            print(f"\nProfile: {stats.perf_data_path} [caller={caller}]")
            print(f"Samples under '{caller}': {under_caller}/{total} ({100*under_caller/max(1,total):.1f}%)\n")

            if category:
                print(f"Filtering by category: {category}")
            if filter:
                print(f"Filtering by: '{filter}'")

            print(f"{'%':>6}  {'Symbol':<50}")
            print("-" * 60)

            count = 0
            for sym, sym_count in sorted(top_symbols.items(), key=lambda x: -x[1]):
                pct = 100 * sym_count / max(1, under_caller)

                # Apply filters
                if filter and filter.lower() not in sym.lower():
                    continue
                if category:
                    # Create temp sample to check category
                    temp = ProfileSample(percent=pct, symbol=sym, library="")
                    if temp.get_category() != category:
                        continue

                print(f"{pct:>5.1f}%  {sym[:50]}")
                count += 1
                if count >= limit:
                    break

            if count == 0:
                print("  (no matching samples after filtering)")
            return

        stats.print_stats(filter=filter, category=category, limit=limit)

    def print_summary(self) -> None:
        """Print high-level summary by category."""
        if self._stats:
            self._stats.print_summary()
        else:
            print("No profiling data available")

    def filter_by_caller(self, caller: str) -> Optional[ProfileStats]:
        """
        Get samples that occurred under a specific caller function.

        Args:
            caller: Function name or pattern to match in call stacks

        Returns:
            New ProfileStats with filtered samples, or None if no data
        """
        if not self._stats:
            return None
        return self._stats.filter_by_caller(caller)

    def analyze_caller(self, caller: str) -> None:
        """
        Analyze and print statistics for samples under a specific caller.

        This directly parses perf script output for more reliable results.

        Args:
            caller: Function name or pattern to match in call stacks
        """
        if not self._stats:
            print("No profiling data available")
            return

        result = self._stats.analyze_by_caller(caller)
        total = result.get("total_samples", 0)
        under = result.get("under_caller", 0)
        symbols = result.get("top_symbols", {})

        print(f"\nCaller Analysis: {caller}")
        print("=" * 50)
        print(f"Total samples:    {total}")
        print(f"Under '{caller}': {under} ({100*under/max(1,total):.1f}%)")

        if symbols:
            print(f"\nTop symbols under {caller}:")
            print("-" * 40)
            sorted_syms = sorted(symbols.items(), key=lambda x: -x[1])[:15]
            for sym, count in sorted_syms:
                pct = 100 * count / max(1, under)
                print(f"  {pct:>5.1f}%  {sym}")

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
