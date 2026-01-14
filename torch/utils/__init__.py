# mypy: allow-untyped-defs

import copyreg
import os.path as _osp
import weakref

import torch
from torch.utils import (
    backcompat as backcompat,
    collect_env as collect_env,
    data as data,
    deterministic as deterministic,
    hooks as hooks,
)
from torch.utils.backend_registration import (
    generate_methods_for_privateuse1_backend,
    rename_privateuse1_backend,
)
from torch.utils.cpp_backtrace import get_cpp_backtrace
from torch.utils.throughput_benchmark import ThroughputBenchmark


def set_module(obj, mod):
    """
    Set the module attribute on a python object for a given object for nicer printing
    """
    if not isinstance(mod, str):
        raise TypeError("The mod argument should be a string")
    obj.__module__ = mod


cmake_prefix_path = _osp.join(_osp.dirname(_osp.dirname(__file__)), "share", "cmake")


def _has_non_weakidref_weakrefs(obj) -> bool:
    """
    Check if obj has any weakrefs that are NOT WeakIdRef.

    WeakIdRef uses id() for identity, which doesn't change during swap_tensors,
    so those weakrefs are safe and don't need to be cleared. Other weakref types
    may not be safe and should prevent swapping.
    """
    from torch.utils.weak import WeakIdRef

    for wr in weakref.getweakrefs(obj):
        if type(wr) is not WeakIdRef:
            return True
    return False


def swap_tensors(t1, t2):
    """
    This function swaps the content of the two Tensor objects.
    At a high level, this will make t1 have the content of t2 while preserving
    its identity.

    This will not work if t1 and t2 have different slots.
    """
    # WeakIdRef weakrefs (from TracingContext.tensor_to_context,
    # MetaTensorDescriber.lookup_tensor, etc.) are safe because they use id()
    # for identity, which doesn't change during swap. Other weakref types
    # may not be safe.
    if _has_non_weakidref_weakrefs(t1):
        raise RuntimeError("Cannot swap t1 because it has weakref associated with it")

    if _has_non_weakidref_weakrefs(t2):
        raise RuntimeError("Cannot swap t2 because it has weakref associated with it")
    t1_slots = set(copyreg._slotnames(t1.__class__))  # type: ignore[attr-defined]
    t2_slots = set(copyreg._slotnames(t2.__class__))  # type: ignore[attr-defined]
    if t1_slots != t2_slots:
        raise RuntimeError("Cannot swap t1 and t2 if they have different slots")

    def swap_attr(name):
        tmp = getattr(t1, name)
        setattr(t1, name, (getattr(t2, name)))
        setattr(t2, name, tmp)

    def error_pre_hook(grad_outputs):
        raise RuntimeError(
            "Trying to execute AccumulateGrad node that was poisoned by swap_tensors "
            "this can happen when you try to run backward on a tensor that was swapped. "
            "For a module m with `torch.__future__.set_swap_module_params_on_conversion(True)` "
            "you should not change the device or dtype of the module (e.g. `m.cpu()` or `m.half()`) "
            "between running forward and backward. To resolve this, please only change the "
            "device/dtype before running forward (or after both forward and backward)."
        )

    def check_use_count(t, name="t1"):
        use_count = t._use_count()
        error_str = (
            f"Expected use_count of {name} to be 1 or 2 with an AccumulateGrad node but got {use_count} "
            f"make sure you are not holding references to the tensor in other places."
        )
        if use_count > 1:
            if use_count == 2 and t.is_leaf:
                accum_grad_node = torch.autograd.graph.get_gradient_edge(t).node
                # Make sure that the accumulate_grad node was not lazy_init-ed by get_gradient_edge
                if t._use_count() == 2:
                    accum_grad_node.register_prehook(error_pre_hook)
                else:
                    raise RuntimeError(error_str)
            else:
                raise RuntimeError(error_str)

    check_use_count(t1, "t1")
    check_use_count(t2, "t2")

    # Swap the types
    # Note that this will fail if there are mismatched slots
    swap_attr("__class__")

    # Swap the dynamic attributes
    swap_attr("__dict__")

    # Swap the slots
    for slot in t1_slots:
        if hasattr(t1, slot) and hasattr(t2, slot):
            swap_attr(slot)
        elif hasattr(t1, slot):
            setattr(t2, slot, (getattr(t1, slot)))
            delattr(t1, slot)
        elif hasattr(t2, slot):
            setattr(t1, slot, (getattr(t2, slot)))
            delattr(t2, slot)

    # Swap the at::Tensor they point to
    torch._C._swap_tensor_impl(t1, t2)

    # Swap tensor_to_context entries in TracingContext if active.
    # This ensures that if t1 had a symbolic context for its OLD shape,
    # it now gets t2's symbolic context (which matches t1's NEW shape).
    # See note [Tensor Fakification and Symbol Caching] in symbolic_shapes.py.
    _swap_tensor_to_context(t1, t2)


def _swap_tensor_to_context(t1, t2):
    """
    Swap the tensor_to_context entries for t1 and t2 in the active TracingContext.

    After swap_tensors(t1, t2), t1 has t2's metadata and vice versa. The symbolic
    context stored in tensor_to_context should also be swapped to maintain the
    invariant that tensor_to_context[tensor] returns the context matching the
    tensor's current shape.
    """
    try:
        tracing_context = torch._guards.TracingContext.try_get()
    except Exception:
        # No tracing context available
        return

    if tracing_context is None:
        return

    tensor_to_context = tracing_context.tensor_to_context

    # Get existing contexts (may be None if tensor was never traced)
    ctx1 = tensor_to_context.get(t1)
    ctx2 = tensor_to_context.get(t2)

    # Swap the contexts
    if ctx1 is not None and ctx2 is not None:
        # Both have contexts - swap them
        tensor_to_context[t1] = ctx2
        tensor_to_context[t2] = ctx1
    elif ctx1 is not None:
        # Only t1 had context - move it to t2, remove from t1
        tensor_to_context[t2] = ctx1
        del tensor_to_context[t1]
    elif ctx2 is not None:
        # Only t2 had context - move it to t1, remove from t2
        tensor_to_context[t1] = ctx2
        del tensor_to_context[t2]
    # If neither has context, nothing to do
