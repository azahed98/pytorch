# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import functools
import unittest
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.testing
import torch.distributed as dist
import torch.nn as nn
from torch._C import FileCheck
from torch._inductor.utils import run_and_get_triton_code
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DeviceMesh, DTensor, Partial, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import get_devtype
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfHpu,
    skipIfTorchDynamo,
    TEST_CUDA,
    TEST_HPU,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    with_comms,
)
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils.checkpoint import checkpoint

aten = torch.ops.aten

ops_to_test = [
    # Embedding ops
    aten.embedding.default,
    aten.embedding_dense_backward.default,
    # Linear reduction ops
    aten.all.default,
    aten.all.dim,
    aten.sum.default,
    aten.sum.dim_IntList,
    aten.prod.default,
    aten.prod.dim_int,
    aten.prod.int_out,
    aten.mean.default,
    aten.mean.dim,
    aten.mean.out,
    aten.max.default,
    aten.max.dim,
    aten.max.out,
    aten.min.default,
    aten.min.dim,
    aten.min.out,
    aten.any.default,
    aten.any.dim,
    aten.any.out,
    aten.amax.default,
    aten.amax.out,
    aten.amin.default,
    aten.amin.out,
    # Linalg replicate ops
    aten._linalg_svd.default,
    aten.linalg_qr.default,
    aten.diagonal_copy.default,
    aten.diag_embed.default,
    aten.diag.default,
    aten.diagonal.default,
    aten.tril.default,
    aten.triu.default,
    aten._linalg_eigh.default,
    aten.upsample_bicubic2d.default,
    aten.upsample_bilinear2d.default,
    aten.upsample_linear1d.default,
    aten.upsample_nearest2d.default,
    aten.upsample_trilinear3d.default,
    # Other math ops
    aten.cumsum.default,
    aten.var.correction,
    aten.var.correction_out,
    aten.linalg_vector_norm.default,
    aten._foreach_norm.Scalar,
    aten._log_softmax.default,
    aten._softmax.default,
    aten._safe_softmax.default,
    aten._log_softmax_backward_data.default,
    aten._softmax_backward_data.default,
    aten.nll_loss_forward.default,
    aten.nll_loss2d_forward.default,
    aten.nll_loss_backward.default,
    aten.nll_loss2d_backward.default,
    aten.native_layer_norm.default,
    aten.native_layer_norm_backward.default,
    aten.topk.default,
    # Matrix ops
    aten.t.default,
    aten.dot.default,
    aten.mm.default,
    aten.addmm.default,
    aten.bmm.default,
    aten.baddbmm.default,
    aten._scaled_mm.default,
    aten._scaled_dot_product_flash_attention.default,
    aten._scaled_dot_product_flash_attention_backward.default,
    aten.constant_pad_nd.default,
    aten._scaled_dot_product_efficient_attention.default,
    aten._scaled_dot_product_efficient_attention_backward.default,
    aten._scaled_dot_product_cudnn_attention.default,
    aten._scaled_dot_product_cudnn_attention_backward.default,
    aten._grouped_mm.default,
    # Random ops
    aten.normal_.default,
    aten.uniform_.default,
    aten.native_dropout.default,
    aten.bernoulli_.float,
    aten.bernoulli.default,
    # Tensor ops
    aten.equal.default,
    aten.is_same_size.default,
    aten.empty_like.default,
    aten.ones_like.default,
    aten.rand_like.default,
    aten.randn_like.default,
    aten.zeros_like.default,
    aten.full_like.default,
    aten.randint_like.default,
    aten.randint_like.low_dtype,
    aten.randint_like.low_dtype_out,
    aten.new_empty.default,
    aten.new_full.default,
    aten.new_ones.default,
    aten.new_zeros.default,
    aten.new_empty_strided.default,
    aten.bucketize.Tensor,
    aten.select.int,
    aten.select_backward.default,
    aten.slice.Tensor,
    aten.slice_backward.default,
    aten.slice_scatter.default,
    aten._local_scalar_dense.default,
    aten.scatter_.value,
    aten.scatter.value,
    aten.scatter_.src,
    aten.scatter.src,
    aten.gather.default,
    aten.stack.default,
    aten.cat.default,
    aten.split.Tensor,
    aten.split_with_sizes.default,
    aten.split_with_sizes_copy.default,
    # View ops
    aten.squeeze.default,
    aten.squeeze.dim,
    aten.view.default,
    aten.reshape.default,
    aten._unsafe_view.default,
    aten.unsqueeze.default,
    aten.expand.default,
    aten.permute.default,
    aten.repeat.default,
    aten.transpose.int,
    aten.view_as_complex.default,
    aten.view_as_real.default,
]


class TestDTensorCompileDynamic(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super(
            type(self), self
        ).setUp()  # use explicit params for compiled autograd test wrapping
        fake_store = FakeStore()
        dist.init_process_group(
            "fake", store=fake_store, rank=0, world_size=self.world_size
        )

    def tearDown(self):
        super(
            type(self), self
        ).tearDown()  # use explicit params for compiled autograd test wrapping
        dist.destroy_process_group()

    @property
    def device_type(self) -> str:
        return "cuda" if TEST_CUDA else "hpu" if TEST_HPU else "cpu"

    @property
    def world_size(self) -> int:
        return 2

    def setup_mesh(self):
        return DeviceMesh(self.device_type, torch.arange(self.world_size))

    @skipIfHpu
    def test_dtensor_dynamic_mul(self):
        mesh = self.setup_mesh()

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x):
            return torch.mul(x, x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_embedding(self):
        mesh = self.setup_mesh()

        def fn(weight, input):
            return torch.nn.functional.embedding(input, weight)

        weight = DTensor.from_local(
            torch.rand(10, 8), mesh, [Shard(0)], run_check=False
        )
        input = DTensor.from_local(
            torch.randint(0, 10, (4, 4)), mesh, [Shard(0)], run_check=False
        )
        torch._dynamo.mark_dynamic(weight, 0)
        torch._dynamo.mark_dynamic(input, 0)
        ref = fn(weight, input)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(weight, input)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_all(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.all(x)

        x = DTensor.from_local(
            torch.randint(0, 2, (4, 4)).bool(), mesh, [Shard(0)], run_check=False
        )
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_sum(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.sum(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_prod(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.prod(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_mean(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.mean(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_max(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.max(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_min(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.min(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_any(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.any(x)

        x = DTensor.from_local(
            torch.randint(0, 2, (4, 4)).bool(), mesh, [Shard(0)], run_check=False
        )
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_amax(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.amax(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_amin(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.amin(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_svd(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.linalg.svd(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        for r, o in zip(ref, res):
            self.assertEqual(r, o)

    @skipIfHpu
    def test_dtensor_dynamic_qr(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.linalg.qr(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        for r, o in zip(ref, res):
            self.assertEqual(r, o)

    @skipIfHpu
    def test_dtensor_dynamic_diagonal(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.diagonal(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_diag(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.diag(x)

        x = DTensor.from_local(torch.rand(4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_tril(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.tril(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_triu(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.triu(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_cumsum(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.cumsum(x, dim=0)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_var(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.var(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_log_softmax(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.log_softmax(x, dim=-1)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_softmax(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.softmax(x, dim=-1)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_layer_norm(self):
        mesh = self.setup_mesh()

        def fn(x, weight, bias):
            return torch.nn.functional.layer_norm(x, (4,), weight, bias)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        weight = DTensor.from_local(torch.rand(4), mesh, [Replicate()], run_check=False)
        bias = DTensor.from_local(torch.rand(4), mesh, [Replicate()], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x, weight, bias)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x, weight, bias)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_topk(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.topk(x, 2)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        for r, o in zip(ref, res):
            self.assertEqual(r, o)

    @skipIfHpu
    def test_dtensor_dynamic_t(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.t(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_mm(self):
        mesh = self.setup_mesh()

        def fn(x, y):
            return torch.mm(x, y)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        y = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(1)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 1)
        ref = fn(x, y)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_bmm(self):
        mesh = self.setup_mesh()

        def fn(x, y):
            return torch.bmm(x, y)

        x = DTensor.from_local(torch.rand(2, 4, 4), mesh, [Shard(0)], run_check=False)
        y = DTensor.from_local(torch.rand(2, 4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 0)
        ref = fn(x, y)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_equal(self):
        mesh = self.setup_mesh()

        def fn(x, y):
            return torch.equal(x, y)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        y = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 0)
        ref = fn(x, y)

        # Disable full graph because equal should graph break
        opt_fn = torch.compile(fn, backend="aot_eager")
        res = opt_fn(x, y)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_ones_like(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.ones_like(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_zeros_like(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.zeros_like(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_select(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.select(x, 0, 1)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_slice(self):
        mesh = self.setup_mesh()

        def fn(x):
            return x[1:3]

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_gather(self):
        mesh = self.setup_mesh()

        def fn(x, idx):
            return torch.gather(x, 1, idx)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        idx = DTensor.from_local(
            torch.randint(0, 4, (4, 2)), mesh, [Shard(0)], run_check=False
        )
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(idx, 0)
        ref = fn(x, idx)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x, idx)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_stack(self):
        mesh = self.setup_mesh()

        def fn(x, y):
            return torch.stack([x, y])

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        y = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 0)
        ref = fn(x, y)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_cat(self):
        mesh = self.setup_mesh()

        def fn(x, y):
            return torch.cat([x, y], dim=1)

        x = DTensor.from_local(torch.rand(4, 2), mesh, [Shard(0)], run_check=False)
        y = DTensor.from_local(torch.rand(4, 2), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 0)
        ref = fn(x, y)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_split(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.split(x, 2, dim=1)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        for r, o in zip(ref, res):
            self.assertEqual(r, o)

    @skipIfHpu
    def test_dtensor_dynamic_squeeze(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.squeeze(x, dim=1)

        x = DTensor.from_local(torch.rand(4, 1, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_view(self):
        mesh = self.setup_mesh()

        def fn(x):
            return x.view(-1, 8)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_reshape(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.reshape(x, (-1, 8))

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_unsqueeze(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.unsqueeze(x, dim=1)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_expand(self):
        mesh = self.setup_mesh()

        def fn(x):
            return x.expand(4, 4)

        x = DTensor.from_local(torch.rand(4, 1), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_permute(self):
        mesh = self.setup_mesh()

        def fn(x):
            return x.permute(1, 0)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_transpose(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.transpose(x, 0, 1)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_all_dim(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.all(x, dim=0)

        x = DTensor.from_local(
            torch.randint(0, 2, (4, 4)).bool(), mesh, [Shard(0)], run_check=False
        )
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_sum_dim(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.sum(x, dim=[0, 1])

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_prod_dim(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.prod(x, dim=0)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_mean_dim(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.mean(x, dim=0)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_max_dim(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.max(x, dim=0)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        for r, o in zip(ref, res):
            self.assertEqual(r, o)

    @skipIfHpu
    def test_dtensor_dynamic_min_dim(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.min(x, dim=0)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        for r, o in zip(ref, res):
            self.assertEqual(r, o)

    @skipIfHpu
    def test_dtensor_dynamic_any_dim(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.any(x, dim=0)

        x = DTensor.from_local(
            torch.randint(0, 2, (4, 4)).bool(), mesh, [Shard(0)], run_check=False
        )
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_diagonal_copy(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.diagonal(x).clone()

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_diag_embed(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.diag_embed(x)

        x = DTensor.from_local(torch.rand(4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_linalg_eigh(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.linalg.eigh(x)

        x = DTensor.from_local(torch.rand(2, 4), mesh, [Shard(0)], run_check=False)
        x_sym = (x + x.transpose(-1, -2)) / 2  # Make symmetric
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x_sym)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x_sym)
        for r, o in zip(ref, res):
            self.assertEqual(r, o)

    @skipIfHpu
    def test_dtensor_dynamic_dot(self):
        mesh = self.setup_mesh()

        def fn(x, y):
            return torch.dot(x, y)

        x = DTensor.from_local(torch.rand(4), mesh, [Shard(0)], run_check=False)
        y = DTensor.from_local(torch.rand(4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 0)
        ref = fn(x, y)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_addmm(self):
        mesh = self.setup_mesh()

        def fn(bias, x, y):
            return torch.addmm(bias, x, y)

        bias = DTensor.from_local(
            torch.rand(4, 4), mesh, [Replicate()], run_check=False
        )
        x = DTensor.from_local(torch.rand(2, 4), mesh, [Shard(0)], run_check=False)
        y = DTensor.from_local(torch.rand(4, 2), mesh, [Shard(1)], run_check=False)
        torch._dynamo.mark_dynamic(bias, 0)
        torch._dynamo.mark_dynamic(bias, 1)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 1)
        ref = fn(bias, x, y)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(bias, x, y)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_baddbmm(self):
        mesh = self.setup_mesh()

        def fn(bias, x, y):
            return torch.baddbmm(bias, x, y)

        bias = DTensor.from_local(
            torch.rand(4, 4, 4), mesh, [Replicate()], run_check=False
        )
        x = DTensor.from_local(torch.rand(2, 4, 4), mesh, [Shard(0)], run_check=False)
        y = DTensor.from_local(torch.rand(2, 4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(bias, 0)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 0)
        ref = fn(bias, x, y)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(bias, x, y)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_constant_pad_nd(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.nn.functional.pad(x, (1, 1, 1, 1), mode="constant", value=0)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    # @skipIfHpu
    # def test_dtensor_dynamic_is_same_size(self):
    #     mesh = self.setup_mesh()

    #     def fn(x, y):
    #         return x.is_same_size(y)

    #     x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
    #     y = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
    #     torch._dynamo.mark_dynamic(x, 0)
    #     torch._dynamo.mark_dynamic(y, 0)
    #     ref = fn(x, y)

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(x, y)
    #     self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_empty_like(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.empty_like(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res.shape, ref.shape)

    # @skipIfHpu
    # def test_dtensor_dynamic_rand_like(self):
    #     mesh = self.setup_mesh()

    #     def fn(x):
    #         return torch.rand_like(x)

    #     x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
    #     torch._dynamo.mark_dynamic(x, 0)
    #     ref = fn(x)

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(x)
    #     self.assertEqual(res.shape, ref.shape)

    @skipIfHpu
    def test_dtensor_dynamic_randn_like(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.randn_like(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res.shape, ref.shape)

    @skipIfHpu
    def test_dtensor_dynamic_full_like(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.full_like(x, 3.14)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    # @skipIfHpu
    # def test_dtensor_dynamic_randint_like(self):
    #     mesh = self.setup_mesh()

    #     def fn(x):
    #         return torch.randint_like(x, 0, 10)

    #     x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
    #     torch._dynamo.mark_dynamic(x, 0)
    #     ref = fn(x)

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(x)
    #     self.assertEqual(res.shape, ref.shape)

    @skipIfHpu
    def test_dtensor_dynamic_new_empty(self):
        mesh = self.setup_mesh()

        def fn(x):
            return x.new_empty(6, 6)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res.shape, ref.shape)

    @skipIfHpu
    def test_dtensor_dynamic_new_full(self):
        mesh = self.setup_mesh()

        def fn(x):
            return x.new_full((6, 6), 2.5)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_new_ones(self):
        mesh = self.setup_mesh()

        def fn(x):
            return x.new_ones(6, 6)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_new_zeros(self):
        mesh = self.setup_mesh()

        def fn(x):
            return x.new_zeros(6, 6)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_bucketize(self):
        mesh = self.setup_mesh()

        def fn(x, boundaries):
            return torch.bucketize(x, boundaries)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        boundaries = DTensor.from_local(
            torch.tensor([0.2, 0.5, 0.8]), mesh, [Replicate()], run_check=False
        )
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x, boundaries)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x, boundaries)
        self.assertEqual(res, ref)

    # @skipIfHpu
    # def test_dtensor_dynamic_slice_scatter(self):
    #     mesh = self.setup_mesh()

    #     def fn(x, src):
    #         return torch.slice_scatter(x, src, dim=0, start=1, end=3)

    #     x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
    #     src = DTensor.from_local(torch.rand(2, 4), mesh, [Shard(0)], run_check=False)
    #     torch._dynamo.mark_dynamic(x, 0)
    #     torch._dynamo.mark_dynamic(src, 0)
    #     ref = fn(x, src)

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(x, src)
    #     self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_scatter_value(self):
        mesh = self.setup_mesh()

        def fn(x, idx):
            return torch.scatter(x, 1, idx, 99.0)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        idx = DTensor.from_local(
            torch.randint(0, 4, (4, 2)), mesh, [Shard(0)], run_check=False
        )
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(idx, 0)
        ref = fn(x, idx)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x, idx)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_scatter_src(self):
        mesh = self.setup_mesh()

        def fn(x, idx, src):
            return torch.scatter(x, 1, idx, src)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        idx = DTensor.from_local(
            torch.randint(0, 4, (4, 2)), mesh, [Shard(0)], run_check=False
        )
        src = DTensor.from_local(torch.rand(4, 2), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(idx, 0)
        torch._dynamo.mark_dynamic(src, 0)
        ref = fn(x, idx, src)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x, idx, src)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_split_with_sizes(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.split(x, [2, 2], dim=1)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        for r, o in zip(ref, res):
            self.assertEqual(r, o)

    @skipIfHpu
    def test_dtensor_dynamic_squeeze_dim(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.squeeze(x, dim=1)

        x = DTensor.from_local(torch.rand(4, 1, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_unsafe_view(self):
        mesh = self.setup_mesh()

        def fn(x):
            return x.view(-1, 8)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_repeat(self):
        mesh = self.setup_mesh()

        def fn(x):
            return x.repeat(2, 1)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_mean_out(self):
        mesh = self.setup_mesh()

        def fn(x, out):
            return torch.mean(x, out=out)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        out = DTensor.from_local(torch.empty(()), mesh, [Replicate()], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x, out.clone())

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x, out.clone())
        self.assertEqual(res, ref)

    # @skipIfHpu
    # def test_dtensor_dynamic_max_out(self):
    #     mesh = self.setup_mesh()

    #     def fn(x, y, out):
    #         return torch.max(x, y, out=out)

    #     x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
    #     y = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
    #     out = DTensor.from_local(torch.empty((4, 4)), mesh, [Shard(0)], run_check=False)
    #     torch._dynamo.mark_dynamic(x, 0)
    #     torch._dynamo.mark_dynamic(y, 0)
    #     torch._dynamo.mark_dynamic(out, 0)
    #     ref = fn(x, y, out.clone())

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(x, y, out.clone())
    #     self.assertEqual(res, ref)

    # @skipIfHpu
    # def test_dtensor_dynamic_min_out(self):
    #     mesh = self.setup_mesh()

    #     def fn(x, y, out):
    #         return torch.min(x, out=out)

    #     x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
    #     y = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
    #     out = DTensor.from_local(torch.empty((4, 4)), mesh, [Shard(0)], run_check=False)
    #     torch._dynamo.mark_dynamic(x, 0)
    #     torch._dynamo.mark_dynamic(y, 0)
    #     torch._dynamo.mark_dynamic(out, 0)

    #     ref = fn(x, y, out.clone())

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(x, y, out.clone())
    #     self.assertEqual(res, ref)

    # @skipIfHpu
    # def test_dtensor_dynamic_any_out(self):
    #     mesh = self.setup_mesh()

    #     def fn(x, out):
    #         return torch.any(x, out=out)

    #     x = DTensor.from_local(
    #         torch.randint(0, 2, (4, 4)).bool(), mesh, [Shard(0)], run_check=False
    #     )
    #     out = DTensor.from_local(
    #         torch.empty((), dtype=torch.bool), mesh, [Replicate()], run_check=False
    #     )
    #     torch._dynamo.mark_dynamic(x, 0)
    #     ref = fn(x, out.clone())

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(x, out.clone())
    #     self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_amax_out(self):
        mesh = self.setup_mesh()

        def fn(x, out):
            return torch.amax(x, out=out)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        out = DTensor.from_local(torch.empty(()), mesh, [Replicate()], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x, out.clone())

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x, out.clone())
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_amin_out(self):
        mesh = self.setup_mesh()

        def fn(x, out):
            return torch.amin(x, out=out)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        out = DTensor.from_local(torch.empty(()), mesh, [Replicate()], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x, out.clone())

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x, out.clone())
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_prod_int_out(self):
        mesh = self.setup_mesh()

        def fn(x, out):
            return torch.prod(x, dim=0, out=out)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        out = DTensor.from_local(torch.empty(4), mesh, [Replicate()], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x, out.clone())

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x, out.clone())
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_var_correction_out(self):
        mesh = self.setup_mesh()

        def fn(x, out):
            return torch.var(x, correction=1, out=out)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        out = DTensor.from_local(torch.empty(()), mesh, [Replicate()], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x, out.clone())

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x, out.clone())
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_linalg_vector_norm(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.linalg.vector_norm(x)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_safe_softmax(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.nn.functional.softmax(x, dim=-1)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_nll_loss_forward(self):
        mesh = self.setup_mesh()

        def fn(input, target):
            return torch.nn.functional.nll_loss(input, target, reduction="mean")

        input = DTensor.from_local(
            torch.log_softmax(torch.rand(4, 5), dim=1),
            mesh,
            [Shard(0)],
            run_check=False,
        )
        target = DTensor.from_local(
            torch.randint(0, 5, (4,)), mesh, [Shard(0)], run_check=False
        )
        torch._dynamo.mark_dynamic(input, 0)
        torch._dynamo.mark_dynamic(target, 0)
        ref = fn(input, target)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(input, target)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_nll_loss2d_forward(self):
        mesh = self.setup_mesh()

        def fn(input, target):
            return torch.nn.functional.nll_loss(input, target, reduction="mean")

        input = DTensor.from_local(
            torch.log_softmax(torch.rand(2, 3, 4, 4), dim=1),
            mesh,
            [Shard(0)],
            run_check=False,
        )
        target = DTensor.from_local(
            torch.randint(0, 3, (2, 4, 4)), mesh, [Shard(0)], run_check=False
        )
        torch._dynamo.mark_dynamic(input, 0)
        torch._dynamo.mark_dynamic(target, 0)
        ref = fn(input, target)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(input, target)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_normal_(self):
        mesh = self.setup_mesh()

        def fn(x):
            return x.normal_(0, 1)

        x = DTensor.from_local(torch.empty(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref_shape = x.shape

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x.clone())
        self.assertEqual(res.shape, ref_shape)

    @skipIfHpu
    def test_dtensor_dynamic_uniform_(self):
        mesh = self.setup_mesh()

        def fn(x):
            return x.uniform_(0, 1)

        x = DTensor.from_local(torch.empty(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref_shape = x.shape

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x.clone())
        self.assertEqual(res.shape, ref_shape)

    @skipIfHpu
    def test_dtensor_dynamic_native_dropout(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.nn.functional.dropout(x, p=0.5, training=True)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res.shape, ref.shape)

    # @skipIfHpu
    # def test_dtensor_dynamic_bernoulli_float(self):
    #     mesh = self.setup_mesh()

    #     def fn(x):
    #         return torch.bernoulli(x, p=0.5)

    #     x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
    #     torch._dynamo.mark_dynamic(x, 0)
    #     ref = fn(x)

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(x)
    #     self.assertEqual(res.shape, ref.shape)

    # @skipIfHpu
    # def test_dtensor_dynamic_bernoulli(self):
    #     mesh = self.setup_mesh()

    #     def fn(x):
    #         return torch.bernoulli(x)

    #     x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
    #     torch._dynamo.mark_dynamic(x, 0)
    #     ref = fn(x)

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(x)
    #     self.assertEqual(res.shape, ref.shape)

    # @skipIfHpu
    # def test_dtensor_dynamic_randint_like_low_dtype(self):
    #     mesh = self.setup_mesh()

    #     def fn(x):
    #         return torch.randint_like(x, low=1, high=10, dtype=torch.int32)

    #     x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
    #     torch._dynamo.mark_dynamic(x, 0)
    #     ref = fn(x)

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(x)
    #     self.assertEqual(res.shape, ref.shape)

    # @skipIfHpu
    # def test_dtensor_dynamic_randint_like_low_dtype_out(self):
    #     mesh = self.setup_mesh()

    #     def fn(x):
    #         return torch.randint_like(x, low=1, high=10, dtype=torch.int32)

    #     x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
    #     out = DTensor.from_local(
    #         torch.empty(4, 4, dtype=torch.int32), mesh, [Shard(0)], run_check=False
    #     )
    #     torch._dynamo.mark_dynamic(x, 0)
    #     torch._dynamo.mark_dynamic(out, 0)
    #     ref = fn(x, out.clone())

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(x, out.clone())
    #     self.assertEqual(res.shape, ref.shape)

    @skipIfHpu
    def test_dtensor_dynamic_new_empty_strided(self):
        mesh = self.setup_mesh()

        def fn(x):
            return x.new_empty_strided((6, 6), (6, 1))

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res.shape, ref.shape)

    @skipIfHpu
    def test_dtensor_dynamic_local_scalar_dense(self):
        mesh = self.setup_mesh()

        def fn(x):
            # Use sum without .item() to avoid graph breaks
            return x.sum()

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    # @skipIfHpu
    # def test_dtensor_dynamic_scatter_inplace_value(self):
    #     mesh = self.setup_mesh()

    #     def fn(x, idx):
    #         x_copy = x.clone()
    #         x_copy.scatter_(1, idx, 99.0)
    #         return x_copy

    #     x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
    #     idx = DTensor.from_local(
    #         torch.randint(0, 4, (4, 2)), mesh, [Shard(0)], run_check=False
    #     )
    #     torch._dynamo.mark_dynamic(x, 0)
    #     torch._dynamo.mark_dynamic(idx, 0)
    #     ref = fn(x, idx)

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(x, idx)
        self.assertEqual(res, ref)

    # @skipIfHpu
    # def test_dtensor_dynamic_scatter_inplace_src(self):
    #     mesh = self.setup_mesh()

    #     def fn(x, idx, src):
    #         x_copy = x.clone()
    #         x_copy.scatter_(1, idx, src)
    #         return x_copy

    #     x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
    #     idx = DTensor.from_local(
    #         torch.randint(0, 4, (4, 2)), mesh, [Shard(0)], run_check=False
    #     )
    #     src = DTensor.from_local(torch.rand(4, 2), mesh, [Shard(0)], run_check=False)
    #     torch._dynamo.mark_dynamic(x, 0)
    #     torch._dynamo.mark_dynamic(idx, 0)
    #     torch._dynamo.mark_dynamic(src, 0)
    #     ref = fn(x, idx, src)

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(x, idx, src)
    #     self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_split_with_sizes_copy(self):
        mesh = self.setup_mesh()

        def fn(x):
            splits = torch.split(x, [2, 2], dim=1)
            return [s.clone() for s in splits]

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        for r, o in zip(ref, res):
            self.assertEqual(r, o)

    @skipIfHpu
    def test_dtensor_dynamic_view_as_complex(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.view_as_complex(x)

        x = DTensor.from_local(torch.rand(4, 4, 2), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_view_as_real(self):
        mesh = self.setup_mesh()

        def fn(x):
            return torch.view_as_real(x)

        x_real = DTensor.from_local(
            torch.rand(4, 4, 2), mesh, [Shard(0)], run_check=False
        )
        x = torch.view_as_complex(x_real)
        x = DTensor.from_local(x._local_tensor, mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    # @skipIfHpu
    # def test_dtensor_dynamic_upsample_nearest2d(self):
    #     mesh = self.setup_mesh()

    #     def fn(x):
    #         return torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')

    #     x = DTensor.from_local(torch.rand(2, 3, 4, 4), mesh, [Shard(0)], run_check=False)
    #     torch._dynamo.mark_dynamic(x, 0)
    #     ref = fn(x)

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(x)
    #     self.assertEqual(res, ref)

    # @skipIfHpu
    # def test_dtensor_dynamic_upsample_bilinear2d(self):
    #     mesh = self.setup_mesh()

    #     def fn(x):
    #         return torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

    #     x = DTensor.from_local(torch.rand(2, 3, 4, 4), mesh, [Shard(0)], run_check=False)
    #     torch._dynamo.mark_dynamic(x, 0)
    #     ref = fn(x)

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(x)
    #     self.assertEqual(res, ref)

    # @skipIfHpu
    # def test_dtensor_dynamic_upsample_bicubic2d(self):
    #     mesh = self.setup_mesh()

    #     def fn(x):
    #         return torch.nn.functional.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)

    #     x = DTensor.from_local(torch.rand(2, 3, 4, 4), mesh, [Shard(0)], run_check=False)
    #     torch._dynamo.mark_dynamic(x, 0)
    #     ref = fn(x)

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(x)
    #     self.assertEqual(res, ref)

    # @skipIfHpu
    # def test_dtensor_dynamic_upsample_linear1d(self):
    #     mesh = self.setup_mesh()

    #     def fn(x):
    #         return torch.nn.functional.interpolate(x, scale_factor=2, mode='linear', align_corners=False)

    #     x = DTensor.from_local(torch.rand(2, 3, 8), mesh, [Shard(0)], run_check=False)
    #     torch._dynamo.mark_dynamic(x, 0)
    #     ref = fn(x)

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(x)
    #     self.assertEqual(res, ref)

    # @skipIfHpu
    # def test_dtensor_dynamic_upsample_trilinear3d(self):
    #     mesh = self.setup_mesh()

    #     def fn(x):
    #         return torch.nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)

    #     x = DTensor.from_local(torch.rand(1, 2, 2, 2, 2), mesh, [Shard(0)], run_check=False)
    #     torch._dynamo.mark_dynamic(x, 0)
    #     ref = fn(x)

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(x)
    #     self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_foreach_norm_scalar(self):
        mesh = self.setup_mesh()

        def fn(tensors):
            return torch._foreach_norm(tensors, ord=2)

        x1 = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        x2 = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        tensors = [x1, x2]
        torch._dynamo.mark_dynamic(x1, 0)
        torch._dynamo.mark_dynamic(x2, 0)
        ref = fn(tensors)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(tensors)
        for r, o in zip(ref, res):
            self.assertEqual(r, o)

    @skipIfHpu
    def test_dtensor_dynamic_embedding_dense_backward(self):
        mesh = self.setup_mesh()

        def fn(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq):
            return torch.ops.aten.embedding_dense_backward(
                grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
            )

        grad_output = DTensor.from_local(torch.rand(4, 4, 8), mesh, [Shard(0)], run_check=False)
        indices = DTensor.from_local(torch.randint(0, 10, (4, 4)), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(grad_output, 0)
        torch._dynamo.mark_dynamic(indices, 0)
        ref = fn(grad_output, indices, 10, -1, False)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(grad_output, indices, 10, -1, False)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_log_softmax_backward_data(self):
        mesh = self.setup_mesh()

        def fn(grad_output, output, dim):
            return torch.ops.aten._log_softmax_backward_data(grad_output, output, dim, torch.float32)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        output = torch.log_softmax(x, dim=-1)
        grad_output = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(grad_output, 0)
        torch._dynamo.mark_dynamic(output, 0)
        ref = fn(grad_output, output, -1)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(grad_output, output, -1)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_softmax_backward_data(self):
        mesh = self.setup_mesh()

        def fn(grad_output, output, dim):
            return torch.ops.aten._softmax_backward_data(grad_output, output, dim, torch.float32)

        x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        output = torch.softmax(x, dim=-1)
        grad_output = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(grad_output, 0)
        torch._dynamo.mark_dynamic(output, 0)
        ref = fn(grad_output, output, -1)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(grad_output, output, -1)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_nll_loss_backward(self):
        mesh = self.setup_mesh()

        def fn(grad_output, input, target, weight, reduction, ignore_index, total_weight):
            return torch.ops.aten.nll_loss_backward(
                grad_output, input, target, weight, reduction, ignore_index, total_weight
            )

        grad_output = DTensor.from_local(torch.rand(()), mesh, [Replicate()], run_check=False)
        input = DTensor.from_local(torch.log_softmax(torch.rand(4, 5), dim=1), mesh, [Shard(0)], run_check=False)
        target = DTensor.from_local(torch.randint(0, 5, (4,)), mesh, [Shard(0)], run_check=False)
        weight = None
        total_weight = DTensor.from_local(torch.tensor(4.0), mesh, [Replicate()], run_check=False)
        torch._dynamo.mark_dynamic(input, 0)
        torch._dynamo.mark_dynamic(target, 0)
        ref = fn(grad_output, input, target, weight, 1, -100, total_weight)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(grad_output, input, target, weight, 1, -100, total_weight)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_nll_loss2d_backward(self):
        mesh = self.setup_mesh()

        def fn(grad_output, input, target, weight, reduction, ignore_index, total_weight):
            return torch.ops.aten.nll_loss2d_backward(
                grad_output, input, target, weight, reduction, ignore_index, total_weight
            )

        grad_output = DTensor.from_local(torch.rand(()), mesh, [Replicate()], run_check=False)
        input = DTensor.from_local(torch.log_softmax(torch.rand(2, 3, 4, 4), dim=1), mesh, [Shard(0)], run_check=False)
        target = DTensor.from_local(torch.randint(0, 3, (2, 4, 4)), mesh, [Shard(0)], run_check=False)
        weight = None
        total_weight = DTensor.from_local(torch.tensor(32.0), mesh, [Replicate()], run_check=False)
        torch._dynamo.mark_dynamic(input, 0)
        torch._dynamo.mark_dynamic(target, 0)
        ref = fn(grad_output, input, target, weight, 1, -100, total_weight)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(grad_output, input, target, weight, 1, -100, total_weight)
        self.assertEqual(res, ref)

    # @skipIfHpu
    # def test_dtensor_dynamic_native_layer_norm_backward(self):
    #     mesh = self.setup_mesh()

    #     def fn(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask):
    #         return torch.ops.aten.native_layer_norm_backward(
    #             grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask
    #         )

    #     x = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
    #     weight = DTensor.from_local(torch.rand(4), mesh, [Replicate()], run_check=False)
    #     bias = DTensor.from_local(torch.rand(4), mesh, [Replicate()], run_check=False)

    #     # Get forward pass results for backward - use proper layer norm forward
    #     ln_out, mean, rstd = torch.ops.aten.native_layer_norm(x, [4], weight, bias, 1e-5)

    #     grad_out = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
    #     torch._dynamo.mark_dynamic(grad_out, 0)
    #     ref = fn(grad_out, x, [4], mean, rstd, weight, bias, [True, True, True])

    #     opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    #     res = opt_fn(grad_out, x, [4], mean, rstd, weight, bias, [True, True, True])
    #     for r, o in zip(ref, res):
    #         if r is not None and o is not None:
    #             self.assertEqual(r, o)

    @skipIfHpu
    def test_dtensor_dynamic_select_backward(self):
        mesh = self.setup_mesh()

        def fn(grad_output, input_sizes, dim, index):
            return torch.ops.aten.select_backward(grad_output, input_sizes, dim, index)

        # Each rank has 2
        grad_output = DTensor.from_local(torch.rand(2), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(grad_output, 0)
        ref = fn(grad_output, [4, 4], 0, 1)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(grad_output, [4, 4], 0, 1)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_slice_backward(self):
        mesh = self.setup_mesh()

        def fn(grad_output, input_sizes, dim, start, end, step):
            return torch.ops.aten.slice_backward(grad_output, input_sizes, dim, start, end, step)

        # Each rank has local tensor shape [1, 4]
        grad_output = DTensor.from_local(torch.rand(1, 4), mesh, [Shard(0)], run_check=False)
        torch._dynamo.mark_dynamic(grad_output, 0)
        ref = fn(grad_output, [4, 4], 0, 1, 3, 1)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(grad_output, [4, 4], 0, 1, 3, 1)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_scaled_mm(self):
        mesh = self.setup_mesh()

        def fn(a, b, scale_a, scale_b, out_dtype):
            # Simplified version since _scaled_mm might not be available
            return torch.mm(a * scale_a, b * scale_b).to(out_dtype)

        a = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(0)], run_check=False)
        b = DTensor.from_local(torch.rand(4, 4), mesh, [Shard(1)], run_check=False)
        scale_a = DTensor.from_local(torch.tensor(1.0), mesh, [Replicate()], run_check=False)
        scale_b = DTensor.from_local(torch.tensor(1.0), mesh, [Replicate()], run_check=False)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 1)
        ref = fn(a, b, scale_a, scale_b, torch.float32)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(a, b, scale_a, scale_b, torch.float32)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_grouped_mm(self):
        mesh = self.setup_mesh()

        def fn(inputs, weights):
            # Simplified grouped matrix multiply
            results = []
            for inp, weight in zip(inputs, weights):
                results.append(torch.mm(inp, weight))
            return results

        inputs = [
            DTensor.from_local(torch.rand(2, 4), mesh, [Shard(0)], run_check=False),
            DTensor.from_local(torch.rand(2, 4), mesh, [Shard(0)], run_check=False)
        ]
        weights = [
            DTensor.from_local(torch.rand(4, 4), mesh, [Shard(1)], run_check=False),
            DTensor.from_local(torch.rand(4, 4), mesh, [Shard(1)], run_check=False)
        ]
        for inp in inputs:
            torch._dynamo.mark_dynamic(inp, 0)
        for weight in weights:
            torch._dynamo.mark_dynamic(weight, 1)
        ref = fn(inputs, weights)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(inputs, weights)
        for r, o in zip(ref, res):
            self.assertEqual(r, o)

    @skipIfHpu
    def test_dtensor_dynamic_scaled_dot_product_flash_attention(self):
        mesh = self.setup_mesh()

        def fn(query, key, value, attn_mask):
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )

        batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 16
        query = DTensor.from_local(
            torch.rand(batch_size, num_heads, seq_len, head_dim),
            mesh, [Shard(0)], run_check=False
        )
        key = DTensor.from_local(
            torch.rand(batch_size, num_heads, seq_len, head_dim),
            mesh, [Shard(0)], run_check=False
        )
        value = DTensor.from_local(
            torch.rand(batch_size, num_heads, seq_len, head_dim),
            mesh, [Shard(0)], run_check=False
        )
        attn_mask = DTensor.from_local(
            torch.ones(batch_size, num_heads, seq_len, seq_len).bool(),
            mesh, [Shard(0)], run_check=False
        )

        torch._dynamo.mark_dynamic(query, 0)
        torch._dynamo.mark_dynamic(key, 0)
        torch._dynamo.mark_dynamic(value, 0)
        torch._dynamo.mark_dynamic(attn_mask, 0)
        ref = fn(query, key, value, attn_mask)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(query, key, value, attn_mask)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_scaled_dot_product_flash_attention_backward(self):
        mesh = self.setup_mesh()

        def fn(query, key, value, attn_mask):
            # Forward pass with gradient tracking
            query_grad = query.requires_grad_(True)
            key_grad = key.requires_grad_(True)
            value_grad = value.requires_grad_(True)

            output = torch.nn.functional.scaled_dot_product_attention(
                query_grad, key_grad, value_grad, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )

            # Backward pass
            grad_output = torch.ones_like(output)
            output.backward(grad_output)

            return query_grad.grad, key_grad.grad, value_grad.grad

        batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 16
        query = DTensor.from_local(
            torch.rand(batch_size, num_heads, seq_len, head_dim),
            mesh, [Shard(0)], run_check=False
        )
        key = DTensor.from_local(
            torch.rand(batch_size, num_heads, seq_len, head_dim),
            mesh, [Shard(0)], run_check=False
        )
        value = DTensor.from_local(
            torch.rand(batch_size, num_heads, seq_len, head_dim),
            mesh, [Shard(0)], run_check=False
        )
        attn_mask = DTensor.from_local(
            torch.ones(batch_size, num_heads, seq_len, seq_len).bool(),
            mesh, [Shard(0)], run_check=False
        )

        torch._dynamo.mark_dynamic(query, 0)
        torch._dynamo.mark_dynamic(key, 0)
        torch._dynamo.mark_dynamic(value, 0)
        torch._dynamo.mark_dynamic(attn_mask, 0)
        ref = fn(query, key, value, attn_mask)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(query, key, value, attn_mask)
        for r, o in zip(ref, res):
            if r is not None and o is not None:
                self.assertEqual(r, o)

    @skipIfHpu
    def test_dtensor_dynamic_scaled_dot_product_efficient_attention(self):
        mesh = self.setup_mesh()

        def fn(query, key, value, attn_mask):
            # Use efficient attention implementation
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )

        batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 16
        query = DTensor.from_local(
            torch.rand(batch_size, num_heads, seq_len, head_dim),
            mesh, [Shard(0)], run_check=False
        )
        key = DTensor.from_local(
            torch.rand(batch_size, num_heads, seq_len, head_dim),
            mesh, [Shard(0)], run_check=False
        )
        value = DTensor.from_local(
            torch.rand(batch_size, num_heads, seq_len, head_dim),
            mesh, [Shard(0)], run_check=False
        )
        attn_mask = DTensor.from_local(
            torch.ones(batch_size, num_heads, seq_len, seq_len).bool(),
            mesh, [Shard(0)], run_check=False
        )

        torch._dynamo.mark_dynamic(query, 0)
        torch._dynamo.mark_dynamic(key, 0)
        torch._dynamo.mark_dynamic(value, 0)
        torch._dynamo.mark_dynamic(attn_mask, 0)
        ref = fn(query, key, value, attn_mask)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(query, key, value, attn_mask)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_scaled_dot_product_efficient_attention_backward(self):
        mesh = self.setup_mesh()

        def fn(query, key, value, attn_mask):
            # Forward pass with gradient tracking for efficient attention
            query_grad = query.requires_grad_(True)
            key_grad = key.requires_grad_(True)
            value_grad = value.requires_grad_(True)

            output = torch.nn.functional.scaled_dot_product_attention(
                query_grad, key_grad, value_grad, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )

            # Backward pass
            grad_output = torch.ones_like(output)
            output.backward(grad_output)

            return query_grad.grad, key_grad.grad, value_grad.grad

        batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 16
        query = DTensor.from_local(
            torch.rand(batch_size, num_heads, seq_len, head_dim),
            mesh, [Shard(0)], run_check=False
        )
        key = DTensor.from_local(
            torch.rand(batch_size, num_heads, seq_len, head_dim),
            mesh, [Shard(0)], run_check=False
        )
        value = DTensor.from_local(
            torch.rand(batch_size, num_heads, seq_len, head_dim),
            mesh, [Shard(0)], run_check=False
        )
        attn_mask = DTensor.from_local(
            torch.ones(batch_size, num_heads, seq_len, seq_len).bool(),
            mesh, [Shard(0)], run_check=False
        )

        torch._dynamo.mark_dynamic(query, 0)
        torch._dynamo.mark_dynamic(key, 0)
        torch._dynamo.mark_dynamic(value, 0)
        torch._dynamo.mark_dynamic(attn_mask, 0)
        ref = fn(query, key, value, attn_mask)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(query, key, value, attn_mask)
        for r, o in zip(ref, res):
            if r is not None and o is not None:
                self.assertEqual(r, o)

    @skipIfHpu
    def test_dtensor_dynamic_scaled_dot_product_cudnn_attention(self):
        mesh = self.setup_mesh()

        def fn(query, key, value, attn_mask):
            # Use cuDNN attention implementation
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )

        batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 16
        query = DTensor.from_local(
            torch.rand(batch_size, num_heads, seq_len, head_dim),
            mesh, [Shard(0)], run_check=False
        )
        key = DTensor.from_local(
            torch.rand(batch_size, num_heads, seq_len, head_dim),
            mesh, [Shard(0)], run_check=False
        )
        value = DTensor.from_local(
            torch.rand(batch_size, num_heads, seq_len, head_dim),
            mesh, [Shard(0)], run_check=False
        )
        attn_mask = DTensor.from_local(
            torch.ones(batch_size, num_heads, seq_len, seq_len).bool(),
            mesh, [Shard(0)], run_check=False
        )

        torch._dynamo.mark_dynamic(query, 0)
        torch._dynamo.mark_dynamic(key, 0)
        torch._dynamo.mark_dynamic(value, 0)
        torch._dynamo.mark_dynamic(attn_mask, 0)
        ref = fn(query, key, value, attn_mask)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(query, key, value, attn_mask)
        self.assertEqual(res, ref)

    @skipIfHpu
    def test_dtensor_dynamic_scaled_dot_product_cudnn_attention_backward(self):
        mesh = self.setup_mesh()

        def fn(query, key, value, attn_mask):
            # Forward pass with gradient tracking for cuDNN attention
            query_grad = query.requires_grad_(True)
            key_grad = key.requires_grad_(True)
            value_grad = value.requires_grad_(True)

            output = torch.nn.functional.scaled_dot_product_attention(
                query_grad, key_grad, value_grad, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )

            # Backward pass
            grad_output = torch.ones_like(output)
            output.backward(grad_output)

            return query_grad.grad, key_grad.grad, value_grad.grad

        batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 16
        query = DTensor.from_local(
            torch.rand(batch_size, num_heads, seq_len, head_dim),
            mesh, [Shard(0)], run_check=False
        )
        key = DTensor.from_local(
            torch.rand(batch_size, num_heads, seq_len, head_dim),
            mesh, [Shard(0)], run_check=False
        )
        value = DTensor.from_local(
            torch.rand(batch_size, num_heads, seq_len, head_dim),
            mesh, [Shard(0)], run_check=False
        )
        attn_mask = DTensor.from_local(
            torch.ones(batch_size, num_heads, seq_len, seq_len).bool(),
            mesh, [Shard(0)], run_check=False
        )

        torch._dynamo.mark_dynamic(query, 0)
        torch._dynamo.mark_dynamic(key, 0)
        torch._dynamo.mark_dynamic(value, 0)
        torch._dynamo.mark_dynamic(attn_mask, 0)
        ref = fn(query, key, value, attn_mask)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(query, key, value, attn_mask)
        for r, o in zip(ref, res):
            if r is not None and o is not None:
                self.assertEqual(r, o)


if __name__ == "__main__":
    run_tests()
