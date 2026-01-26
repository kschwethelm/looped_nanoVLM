import unittest

import torch

from models.config import VLMConfig
from models.language_model import LanguageModel


class TestLoopedTransformer(unittest.TestCase):
    def test_looped_lm_output_shape(self):
        """Verify looped LM produces correct output shape."""
        cfg = VLMConfig(
            lm_looped=True,
            lm_n_prelude_layers=2,
            lm_n_core_layers=4,
            lm_n_loops=7,
            lm_n_coda_layers=2,
            lm_use_tokens=False,  # Test with embeddings
            lm_n_blocks=32,  # For comparison
        )
        model = LanguageModel(cfg)
        model.eval()

        # Test with embedding input
        x = torch.randn(2, 10, cfg.lm_hidden_dim)
        out, kv_cache = model(x)

        # Verify output shape matches input
        self.assertEqual(out.shape, x.shape)

        # Verify KV cache has correct number of entries
        # 2 + 4*7 + 2 = 32 virtual layers
        expected_virtual_layers = (
            cfg.lm_n_prelude_layers + cfg.lm_n_core_layers * cfg.lm_n_loops + cfg.lm_n_coda_layers
        )
        self.assertEqual(len(kv_cache), expected_virtual_layers)
        self.assertEqual(len(kv_cache), 32)

    def test_looped_lm_parameter_count(self):
        """Verify parameter reduction in looped architecture."""
        # Standard model with 32 blocks
        cfg_standard = VLMConfig(lm_looped=False, lm_n_blocks=32, lm_use_tokens=False)

        # Looped model with same effective depth
        cfg_looped = VLMConfig(
            lm_looped=True,
            lm_n_prelude_layers=2,
            lm_n_core_layers=4,
            lm_n_loops=7,
            lm_n_coda_layers=2,
            lm_use_tokens=False,
            lm_n_blocks=32,  # Only used for from_pretrained
        )

        model_standard = LanguageModel(cfg_standard)
        model_looped = LanguageModel(cfg_looped)

        # Count only block parameters (excluding embeddings, norm, head)
        def count_block_params(model):
            if hasattr(model, "blocks"):
                return sum(p.numel() for p in model.blocks.parameters())
            else:
                return (
                    sum(p.numel() for p in model.prelude_blocks.parameters())
                    + sum(p.numel() for p in model.core_blocks.parameters())
                    + sum(p.numel() for p in model.coda_blocks.parameters())
                )

        standard_params = count_block_params(model_standard)
        looped_params = count_block_params(model_looped)

        # Looped should have 8/32 = 25% of standard block params
        # (2 prelude + 4 core + 2 coda = 8 unique blocks vs 32 standard blocks)
        expected_ratio = (
            cfg_looped.lm_n_prelude_layers
            + cfg_looped.lm_n_core_layers
            + cfg_looped.lm_n_coda_layers
        ) / cfg_standard.lm_n_blocks

        actual_ratio = looped_params / standard_params

        print("\nParameter comparison:")
        print(f"  Standard blocks: {standard_params:,} parameters")
        print(f"  Looped blocks: {looped_params:,} parameters")
        print(f"  Reduction: {(1 - actual_ratio) * 100:.1f}%")

        # Verify the parameter count is as expected
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=5)
        self.assertLess(looped_params, standard_params * 0.3)

    def test_kv_cache_generation(self):
        """Verify KV cache works during generation with looped architecture."""
        cfg = VLMConfig(
            lm_looped=True,
            lm_n_prelude_layers=2,
            lm_n_core_layers=4,
            lm_n_loops=7,
            lm_n_coda_layers=2,
            lm_use_tokens=True,
            lm_n_blocks=32,
        )
        model = LanguageModel(cfg)
        model.eval()

        batch_size = 2
        prefill_len = 5

        # Prefill phase
        input_ids = torch.randint(0, cfg.lm_vocab_size, (batch_size, prefill_len))
        output_prefill, kv_cache = model(input_ids, kv_cache=None, start_pos=0)

        # Verify prefill output shape
        self.assertEqual(output_prefill.shape, (batch_size, prefill_len, cfg.lm_vocab_size))

        # Verify KV cache was created for all virtual layers
        expected_virtual_layers = (
            cfg.lm_n_prelude_layers + cfg.lm_n_core_layers * cfg.lm_n_loops + cfg.lm_n_coda_layers
        )
        self.assertEqual(len(kv_cache), expected_virtual_layers)

        # Verify KV cache has correct shape for first layer
        self.assertIsNotNone(kv_cache[0])
        self.assertIn("key", kv_cache[0])
        self.assertIn("value", kv_cache[0])
        self.assertEqual(kv_cache[0]["key"].shape[2], prefill_len)  # Sequence dimension

        # Decode phase - add one token
        new_token = torch.randint(0, cfg.lm_vocab_size, (batch_size, 1))
        output_decode, kv_cache = model(new_token, kv_cache=kv_cache, start_pos=prefill_len)

        # Verify decode output shape
        self.assertEqual(output_decode.shape, (batch_size, 1, cfg.lm_vocab_size))

        # Verify cache grew
        self.assertEqual(kv_cache[0]["key"].shape[2], prefill_len + 1)
        self.assertEqual(kv_cache[0]["value"].shape[2], prefill_len + 1)

    def test_kv_cache_consistency_looped(self):
        """Verify looped model produces consistent outputs with/without KV cache."""
        cfg = VLMConfig(
            lm_looped=True,
            lm_n_prelude_layers=2,
            lm_n_core_layers=4,
            lm_n_loops=7,
            lm_n_coda_layers=2,
            lm_use_tokens=True,
            lm_n_blocks=32,
        )
        model = LanguageModel(cfg)
        model.eval()

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, cfg.lm_vocab_size, (batch_size, seq_len))

        # Forward pass without KV caching (full sequence)
        output_no_cache, _ = model(input_ids, start_pos=0)

        # Forward pass with KV caching (prefill + decode)
        # Prefill with all but last token
        prefill_output, kv_cache = model(input_ids[:, :-1], start_pos=0)

        # Decode last token
        last_token_input = input_ids[:, -1:]
        output_with_cache, _ = model(last_token_input, kv_cache=kv_cache, start_pos=seq_len - 1)

        # Compare the last token's logits
        logits_no_cache = output_no_cache[:, -1, :]
        logits_with_cache = output_with_cache[:, 0, :]

        self.assertTrue(
            torch.allclose(logits_no_cache, logits_with_cache, atol=1e-5),
            "Outputs with and without KV caching do not match for looped model",
        )

    def test_looped_vs_standard_architecture_switch(self):
        """Verify model can be created in both standard and looped modes."""
        # Create standard model
        cfg_standard = VLMConfig(lm_looped=False, lm_n_blocks=32, lm_use_tokens=False)
        model_standard = LanguageModel(cfg_standard)

        # Verify standard model has 'blocks' attribute
        self.assertTrue(hasattr(model_standard, "blocks"))
        self.assertFalse(hasattr(model_standard, "prelude_blocks"))
        self.assertFalse(hasattr(model_standard, "core_blocks"))
        self.assertFalse(hasattr(model_standard, "coda_blocks"))

        # Create looped model
        cfg_looped = VLMConfig(
            lm_looped=True,
            lm_n_prelude_layers=2,
            lm_n_core_layers=4,
            lm_n_loops=7,
            lm_n_coda_layers=2,
            lm_use_tokens=False,
        )
        model_looped = LanguageModel(cfg_looped)

        # Verify looped model has separate block attributes
        self.assertFalse(hasattr(model_looped, "blocks"))
        self.assertTrue(hasattr(model_looped, "prelude_blocks"))
        self.assertTrue(hasattr(model_looped, "core_blocks"))
        self.assertTrue(hasattr(model_looped, "coda_blocks"))

        # Verify block counts
        self.assertEqual(len(model_looped.prelude_blocks), 2)
        self.assertEqual(len(model_looped.core_blocks), 4)
        self.assertEqual(len(model_looped.coda_blocks), 2)


if __name__ == "__main__":
    unittest.main()
