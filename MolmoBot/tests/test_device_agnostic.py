"""Device selection tests for inference. The XPU parity test is skipped without an XPU."""
import unittest
from unittest import mock

import torch

from olmo import device_util
from olmo.nn.llm import LayerNorm, LlmConfig, OLMoBlock
from olmo.torch_util import BufferCache


def _fake_host(xpu: bool, cuda: bool):
    return mock.patch.object(device_util, "is_available", side_effect=lambda b: {"xpu": xpu, "cuda": cuda}.get(b, False))


class TestDeviceSelection(unittest.TestCase):
    def test_default_prefers_xpu_then_cuda_then_cpu(self):
        with _fake_host(xpu=True, cuda=True):
            self.assertEqual(device_util.default_device_type(), "xpu")
        with _fake_host(xpu=False, cuda=True):
            self.assertEqual(device_util.default_device_type(), "cuda")
        with _fake_host(xpu=False, cuda=False):
            self.assertEqual(device_util.default_device_type(), "cpu")

    def test_explicit_device_is_honoured(self):
        with _fake_host(xpu=True, cuda=True):
            self.assertEqual(device_util.resolve_device("cuda"), torch.device("cuda"))
            self.assertEqual(device_util.resolve_device("xpu:1"), torch.device("xpu:1"))
            self.assertEqual(device_util.resolve_device("cpu"), torch.device("cpu"))

    def test_unavailable_backend_falls_back_to_cpu(self):
        with _fake_host(xpu=False, cuda=False):
            self.assertEqual(device_util.resolve_device("cuda"), torch.device("cpu"))
            self.assertEqual(device_util.resolve_device("xpu"), torch.device("cpu"))
            self.assertEqual(device_util.resolve_device(None), torch.device("cpu"))

    def test_device_count_is_zero_without_the_backend(self):
        with _fake_host(xpu=False, cuda=False):
            self.assertEqual(device_util.device_count("xpu"), 0)
            self.assertEqual(device_util.device_count(), 0)



class TestAutocastCasts(unittest.TestCase):
    def test_cpu_layer_norm_casts_are_unchanged(self):
        x = torch.randn(2, 4)
        norm = LayerNorm.build(_tiny_config())
        self.assertIs(norm._cast_if_autocast_enabled(x), x)

    @unittest.skipUnless(torch.xpu.is_available(), "needs an XPU")
    def test_xpu_autocast_casts_to_bfloat16(self):
        # get_autocast_gpu_dtype reports float16 inside an XPU autocast region; the casts must not use it
        norm = LayerNorm.build(_tiny_config())
        x = torch.randn(2, 4, device="xpu")
        with torch.autocast("xpu", dtype=torch.bfloat16):
            self.assertEqual(norm._cast_if_autocast_enabled(x).dtype, torch.bfloat16)
            bias = OLMoBlock._cast_attn_bias(torch.zeros(2, 2, device="xpu"), torch.float32)
            self.assertEqual(bias.dtype, torch.bfloat16)
        self.assertEqual(norm._cast_if_autocast_enabled(x).dtype, torch.float32)


def _tiny_config() -> LlmConfig:
    return LlmConfig(
        d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, n_layers=2, mlp_hidden_size=128,
        activation_type="swiglu", block_type="sequential", rope=True, rope_theta=10000.0,
        attention_type="sdpa", float32_attention=True, attention_layer_norm=True,
        attention_layer_norm_type="qwen3", layer_norm_type="rms", max_sequence_length=64,
        vocab_size=128, embedding_size=128, additional_vocab_size=8, residual_dropout=0.0,
        attention_dropout=0.0, init_std=0.02,
    )


@unittest.skipUnless(torch.xpu.is_available(), "needs an XPU")
class TestBlockParityOnXpu(unittest.TestCase):
    def test_transformer_block_cpu_vs_xpu(self):
        torch.manual_seed(0)
        config = _tiny_config()
        block = OLMoBlock.build(0, config, BufferCache(), "cpu").eval()
        with torch.no_grad():
            for p in block.parameters():  # random weights everywhere, no zero-initialised layer
                p.copy_(torch.randn_like(p) * 0.2)
        x = torch.randn(2, 16, config.d_model)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)
        with torch.no_grad():
            ref, _ = block(x, position_ids=position_ids)
        self.assertGreater(ref.std().item(), 1e-3)

        xpu_block = OLMoBlock.build(0, config, BufferCache(), "cpu").eval()
        xpu_block.load_state_dict(block.state_dict())
        xpu_block = xpu_block.to("xpu", torch.bfloat16)
        with torch.no_grad(), torch.autocast("xpu", dtype=torch.bfloat16):
            out, _ = xpu_block(x.to("xpu", torch.bfloat16), position_ids=position_ids.to("xpu"))
        out = out.float().cpu()
        self.assertTrue(torch.isfinite(out).all())
        rel_mse = ((out - ref) ** 2).sum() / (ref ** 2).sum()
        self.assertLess(rel_mse.item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
