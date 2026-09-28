"""B5 (Part B pre-registration): the false-positive term on a batch with no benign samples.

src/training_treespike.distillation_loss returns a constant zero FP term when the batch
has no benign flow. These tests pin that down: the loss stays finite, the FP term adds
no gradient on such a batch, and it does add one when benign flows are present. The
ranking term draws a random pairing, so the generator is re-seeded before each call and
the two calls compared differ only in a_fp.

    python -m unittest tests.test_fp_term
"""
import unittest

import torch

from src.training_treespike import distillation_loss


def grads(logits, y, teacher, a_fp):
    x = logits.clone().requires_grad_(True)
    torch.manual_seed(0)
    total, parts = distillation_loss(x, y, teacher, torch.tensor(0.1), a_fp=a_fp)
    total.backward()
    return total.detach(), parts, x.grad.detach()


class FalsePositiveTerm(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(1)
        self.logits = torch.randn(32, 5, generator=g)
        self.teacher = torch.softmax(torch.randn(32, 5, generator=g), -1)

    def test_no_benign_batch_is_finite_and_adds_no_gradient(self):
        y = torch.randint(1, 5, (32,))                       # classes 1..4 only: no benign
        total, parts, g_on = grads(self.logits, y, self.teacher, a_fp=0.5)
        _, _, g_off = grads(self.logits, y, self.teacher, a_fp=0.0)
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(parts["fp"], 0.0)
        self.assertTrue(torch.isfinite(g_on).all())
        self.assertTrue(torch.equal(g_on, g_off))           # the FP term contributes nothing

    def test_benign_batch_adds_a_gradient(self):
        y = torch.randint(0, 5, (32,))
        y[:4] = 0                                           # benign flows present
        _, parts, g_on = grads(self.logits, y, self.teacher, a_fp=0.5)
        _, _, g_off = grads(self.logits, y, self.teacher, a_fp=0.0)
        self.assertGreater(parts["fp"], 0.0)
        self.assertGreater((g_on - g_off).abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
