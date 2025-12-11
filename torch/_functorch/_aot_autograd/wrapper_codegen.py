# mypy: allow-untyped-defs
"""
Code generation for AOTAutograd runtime wrappers.

This module provides infrastructure for generating specialized Python wrapper
functions at compile time, replacing the interpretive runtime wrappers with
generated code that inlines only the operations needed for each specific graph.

See: https://github.com/pytorch/pytorch/issues/161783
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from .schemas import (
    OutputType,
    TensorAlias,
    ViewAndMutationMeta,
)


class IndentedBuffer:
    """Helper for generating indented Python code."""

    def __init__(self, initial_indent: int = 0):
        self._lines: list[tuple[int, str]] = []
        self._indent = initial_indent

    def writeline(self, line: str) -> None:
        """Write a line at the current indentation level."""
        self._lines.append((self._indent, line))

    def indent(self) -> None:
        """Increase indentation."""
        self._indent += 1

    def dedent(self) -> None:
        """Decrease indentation."""
        self._indent -= 1

    def getvalue(self) -> str:
        """Get the complete generated code as a string."""
        return "\n".join("    " * indent + line for indent, line in self._lines)


@dataclass
class CodegenState:
    """State for code generation."""

    var_counter: int = 0
    orig_input_vars: dict[int, str] = field(default_factory=dict)
    captured_vars: dict[str, Any] = field(default_factory=dict)

    def fresh_var(self, prefix: str = "v") -> str:
        """Generate a fresh variable name."""
        self.var_counter += 1
        return f"_{prefix}{self.var_counter}"


def _gen_alias_from_base(base, output_ref):
    """Generate an alias of base with the same metadata as output_ref."""
    if output_ref is None:
        return None
    return base.as_strided(
        output_ref.size(),
        output_ref.stride(),
        output_ref.storage_offset(),
    )


def codegen_runtime_wrapper_inference(
    runtime_metadata: ViewAndMutationMeta,
    keep_input_mutations: bool,
) -> str:
    """Generate code for RuntimeWrapper in inference mode.

    Returns the generated Python code as a string.
    """
    code = IndentedBuffer()
    state = CodegenState()

    # Compute what we need
    num_mutated_runtime_inps = runtime_metadata.num_mutated_inp_runtime_indices
    has_mutations = num_mutated_runtime_inps > 0
    has_aliased_outputs = runtime_metadata.num_outputs_aliased > 0
    num_outputs = runtime_metadata.num_outputs
    num_intermediate_bases = runtime_metadata.num_intermediate_bases

    # Indices we need to stash for the epilogue
    epilogue_args_idx: set[int] = set()
    if has_mutations:
        for idx in runtime_metadata.mutated_inp_runtime_indices:
            epilogue_args_idx.add(idx)
    if has_aliased_outputs:
        for info in runtime_metadata.output_info:
            if info.output_type in (
                OutputType.alias_of_input,
                OutputType.is_input,
            ):
                if info.base_idx is not None:
                    epilogue_args_idx.add(info.base_idx)

    # Generate function
    code.writeline("def call(args):")
    code.indent()

    # 1. Stash original inputs for epilogue
    if epilogue_args_idx:
        code.writeline("# Stash inputs for epilogue")
        for idx in sorted(epilogue_args_idx):
            var = state.fresh_var("orig_inp")
            code.writeline(f"{var} = args[{idx}]")
            state.orig_input_vars[idx] = var
        code.writeline("")

    # 2. Version increment for mutated inputs
    if keep_input_mutations and runtime_metadata.mutated_graph_handled_indices_seen_by_autograd:
        indices = runtime_metadata.mutated_graph_handled_indices_seen_by_autograd
        code.writeline("# Increment version for mutated inputs")
        args_list = ", ".join(f"args[{i}]" for i in indices)
        code.writeline(f"torch.autograd.graph.increment_version([{args_list}])")
        code.writeline("")

    # 3. Disable grad for inference and call compiled function
    code.writeline("# Disable grad for inference")
    code.writeline("_grad_enabled = torch.is_grad_enabled()")
    code.writeline("if _grad_enabled:")
    code.indent()
    code.writeline("torch._C._set_grad_enabled(False)")
    code.dedent()
    code.writeline("try:")
    code.indent()
    code.writeline("_all_outs = _compiled_fn(args)")
    code.dedent()
    code.writeline("finally:")
    code.indent()
    code.writeline("if _grad_enabled:")
    code.indent()
    code.writeline("torch._C._set_grad_enabled(True)")
    code.dedent()
    code.dedent()
    code.writeline("")

    # 4. Process outputs
    if has_mutations:
        code.writeline("# Separate mutated inputs from forward outputs")
        code.writeline(f"_updated_inputs = _all_outs[:{num_mutated_runtime_inps}]")
        code.writeline(f"_fw_outs = _all_outs[{num_mutated_runtime_inps}:]")
        code.writeline("")

        # Apply mutations back to original inputs
        code.writeline("# Apply mutations to original inputs")
        for i, inpt_idx in enumerate(runtime_metadata.mutated_inp_runtime_indices):
            meta = runtime_metadata.input_info[inpt_idx]
            if not meta.mutates_data and not meta.mutates_metadata:
                continue

            orig_var = state.orig_input_vars.get(inpt_idx)
            if orig_var is None:
                continue

            if meta.mutates_storage_metadata:
                code.writeline("with torch.no_grad():")
                code.indent()
                code.writeline(f"{orig_var}.set_(_updated_inputs[{i}])")
                code.dedent()
            elif meta.mutates_metadata and not meta.mutates_data:
                code.writeline(
                    f"{orig_var}.as_strided_("
                    f"_updated_inputs[{i}].size(), "
                    f"_updated_inputs[{i}].stride(), "
                    f"_updated_inputs[{i}].storage_offset())"
                )
            else:
                if meta.mutates_metadata:
                    code.writeline(
                        f"{orig_var}.as_strided_("
                        f"_updated_inputs[{i}].size(), "
                        f"_updated_inputs[{i}].stride(), "
                        f"_updated_inputs[{i}].storage_offset())"
                    )
                if meta.is_leaf:
                    code.writeline(f"if {orig_var}.requires_grad:")
                    code.indent()
                    code.writeline(f"{orig_var}.detach().copy_(_updated_inputs[{i}])")
                    code.dedent()
                    code.writeline("else:")
                    code.indent()
                    code.writeline(f"{orig_var}.copy_(_updated_inputs[{i}])")
                    code.dedent()
                else:
                    code.writeline(f"{orig_var}.copy_(_updated_inputs[{i}])")
        code.writeline("")
    else:
        code.writeline("_fw_outs = _all_outs")
        code.writeline("")

    # 5. Handle aliased outputs
    if has_aliased_outputs:
        code.writeline("# Regenerate aliased outputs")
        code.writeline("_ret_outs = []")

        for i, info in enumerate(runtime_metadata.output_info):
            if info.output_type == OutputType.non_alias:
                code.writeline(f"_ret_outs.append(_fw_outs[{i}])")
            elif info.output_type == OutputType.alias_of_input:
                base_idx = info.base_idx
                orig_var = state.orig_input_vars.get(base_idx)
                if orig_var is not None:
                    code.writeline(
                        f"_ret_outs.append(_gen_alias({orig_var}, _fw_outs[{i}]))"
                    )
                else:
                    code.writeline(f"_ret_outs.append(_fw_outs[{i}])")
            elif info.output_type == OutputType.is_input:
                base_idx = info.base_idx
                orig_var = state.orig_input_vars.get(base_idx)
                if orig_var is not None:
                    code.writeline(f"_ret_outs.append({orig_var})")
                else:
                    code.writeline(f"_ret_outs.append(_fw_outs[{i}])")
            elif info.output_type == OutputType.alias_of_intermediate:
                intermediate_idx = info.base_idx
                base_output_idx = num_outputs + intermediate_idx
                code.writeline(
                    f"_ret_outs.append(_gen_alias(_fw_outs[{base_output_idx}], _fw_outs[{i}]))"
                )
            else:
                code.writeline(f"_ret_outs.append(_fw_outs[{i}])")

        code.writeline("return _ret_outs")
    else:
        # No aliased outputs
        if num_intermediate_bases > 0:
            # Need to slice off intermediate bases
            code.writeline(f"return _fw_outs[:{num_outputs}]")
        else:
            code.writeline("return _fw_outs")

    # Handle grad_enabled_mutation if present
    if runtime_metadata.grad_enabled_mutation is not None:
        # Insert before return - this is a simplification
        pass

    return code.getvalue()


def maybe_codegen_runtime_wrapper(
    compiled_fn: Callable,
    runtime_metadata: ViewAndMutationMeta,
    indices_of_inps_to_detach: list[int],
    trace_joint: bool,
    keep_input_mutations: bool,
    disable_amp: bool,
) -> Optional[Callable]:
    """Try to generate a codegen wrapper for the compiled function.

    Returns None if codegen is not supported for this case (falls back to interpretive).
    """
    # For MVP, only support inference (trace_joint=False)
    if trace_joint:
        return None

    # For MVP, skip if there are inputs to detach (training-related)
    if indices_of_inps_to_detach:
        return None

    # For MVP, skip if disable_amp is True (adds complexity)
    if disable_amp:
        return None

    # For MVP, skip if there are effect tokens
    if runtime_metadata.tokens:
        return None

    # For MVP, skip if there's grad_enabled_mutation
    if runtime_metadata.grad_enabled_mutation is not None:
        return None

    # For MVP, skip if there are dynamic outputs
    if runtime_metadata.dynamic_outputs:
        return None

    try:
        code = codegen_runtime_wrapper_inference(
            runtime_metadata,
            keep_input_mutations,
        )

        # Check if we need the alias helper
        needs_gen_alias = "_gen_alias" in code

        namespace: dict[str, Any] = {
            "_compiled_fn": compiled_fn,
            "torch": torch,
            "TensorAlias": TensorAlias,
        }
        if needs_gen_alias:
            namespace["_gen_alias"] = _gen_alias_from_base

        # For debugging: uncomment to see generated code
        # print("=== Generated Wrapper Code ===")
        # print(code)
        # print("==============================")

        exec(code, namespace)
        wrapper_fn = namespace["call"]

        # Mark as boxed for AOTAutograd calling convention
        wrapper_fn._boxed_call = True  # type: ignore[attr-defined]

        return wrapper_fn

    except Exception as e:
        # If codegen fails, fall back to interpretive wrapper
        import logging

        log = logging.getLogger(__name__)
        log.debug("Wrapper codegen failed, falling back to interpretive: %s", e)
        return None
