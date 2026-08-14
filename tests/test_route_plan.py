import importlib
import unittest

import torch


build_route_plan = importlib.import_module(
    "ComfyUI-SolAttn_triton._preprocess"
).build_route_plan


class RoutePlanTests(unittest.TestCase):
    def test_plan_enforces_floor_locality_and_sinks(self):
        blocks, block_size = 6, 64
        q = torch.zeros((1, blocks * block_size, 1, 2), dtype=torch.bfloat16)
        q[..., 0] = 1
        kc = torch.zeros((1, blocks, 1, 2), dtype=torch.bfloat16)
        kc[0, :, 0, 0] = torch.tensor([5, 4, 0, -1, -2, -3])

        mask, counts = build_route_plan(
            q,
            kc,
            tokens=q.shape[1],
            scale=1.0,
            coverage=0.9,
            min_exact_fraction=0.5,
            sink_blocks=(0, 1),
            sink_q=(0, 1),
        )

        self.assertEqual(mask.shape, (1, blocks, 1, blocks))
        self.assertEqual(counts.shape, (1, blocks, 1))
        self.assertTrue(torch.all(counts >= 3))
        self.assertTrue(torch.all(mask[:, :, :, 0] == 1))
        self.assertTrue(torch.all(mask[:, 0, :, :] == 1))
        for query in range(blocks):
            self.assertEqual(mask[0, query, 0, query].item(), 1)
            self.assertEqual(
                int(mask[0, query, 0].sum()), counts[0, query, 0].item()
            )

    def test_plan_handles_ragged_tail(self):
        q = torch.ones((1, 130, 1, 2), dtype=torch.bfloat16)
        kc = torch.ones((1, 3, 1, 2), dtype=torch.bfloat16)
        mask, counts = build_route_plan(
            q,
            kc,
            tokens=130,
            scale=0.5,
            coverage=1.0,
            min_exact_fraction=0.0,
        )
        self.assertTrue(torch.all(mask == 1))
        self.assertTrue(torch.all(counts == 3))

    def test_plan_rejects_invalid_policy(self):
        q = torch.ones((1, 64, 1, 2), dtype=torch.bfloat16)
        kc = torch.ones((1, 1, 1, 2), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "coverage"):
            build_route_plan(
                q, kc, tokens=64, scale=1.0,
                coverage=0.0, min_exact_fraction=0.5,
            )
        with self.assertRaisesRegex(ValueError, "min_exact_fraction"):
            build_route_plan(
                q, kc, tokens=64, scale=1.0,
                coverage=0.9, min_exact_fraction=1.1,
            )


if __name__ == "__main__":
    unittest.main()
