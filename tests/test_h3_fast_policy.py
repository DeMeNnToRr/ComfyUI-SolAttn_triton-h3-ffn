import asyncio
import importlib
import types
import unittest
from unittest import mock


sol_attn = importlib.import_module("ComfyUI-SolAttn_triton")


class MiniMaxH3FastPolicyTests(unittest.TestCase):
    def test_int8_starts_at_measured_crossover(self):
        self.assertFalse(sol_attn._int8_for_tokens(True, 12_287, 12_288))
        self.assertTrue(sol_attn._int8_for_tokens(True, 12_288, 12_288))
        self.assertFalse(sol_attn._int8_for_tokens(False, 20_000, 12_288))
        self.assertTrue(sol_attn._int8_for_tokens(True, 4_096, None))

    def test_h3_fast_node_is_registered(self):
        extension = asyncio.run(sol_attn.comfy_entrypoint())
        names = {node.__name__ for node in asyncio.run(extension.get_node_list())}
        self.assertIn("MiniMaxH3FastPatch", names)

    def test_h3_fast_preset_uses_accepted_int8_routes_for_long_sequences(self):
        diffusion_model = types.SimpleNamespace(
            rope_freqs=object(), _forward=object(), blocks=[object()] * 50
        )
        model = mock.Mock()
        model.get_model_object.return_value = diffusion_model

        self.assertIs(sol_attn.MiniMaxH3FastPatch.execute(model)[0], model)
        with mock.patch.object(sol_attn.SolAttnPatch, "execute", return_value="patched") as execute:
            self.assertEqual(
                sol_attn.MiniMaxH3FastPatch.execute(model, enabled=True), "patched"
            )

        options = execute.call_args.kwargs
        self.assertEqual(options["min_tokens"], 12_288)
        self.assertTrue(options["int8_qk"])
        self.assertTrue(options["int8_pv"])
        self.assertIsNone(options["int8_min_tokens"])
        self.assertEqual(options["route_coverage"], 0.9)
        self.assertEqual(options["min_exact_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
