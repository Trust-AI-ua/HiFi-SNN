"""Validation checkpoint selection and three-precision capture (Part B, amendment A1).

One rule for every trainer: after each epoch, macro-F1 of the model on the loop's own
validation pass (the predictions the loop already computes; no extra forward pass, no
random draw); the best epoch's weights are kept (ties: the earliest epoch) and loaded back
after training, so everything downstream (test scores, int8 projection, checkpoints,
operation counts) comes from the selected checkpoint. Off unless the config sets
``select_on_val: true``; with it off the trainers are unchanged.

``precision_scores`` scores a model at fp32 and at the two int8 projections used in the
paper (per-tensor ``quantize_int8_``, per-channel ``quantize_int8_per_channel_``, both on
every nn.Linear weight), on copies, so the model passed in is not modified.
"""
from __future__ import annotations

import copy

import numpy as np
from sklearn.metrics import f1_score


class ValSelector:
    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self.best, self.epoch, self.state, self.curve = -np.inf, None, None, []

    def update(self, epoch: int, y_true, y_pred, model):
        if not self.enabled:
            return None
        f1 = 100.0 * float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        self.curve.append(f1)
        if f1 > self.best:                      # strict: ties keep the earlier epoch
            self.best, self.epoch = f1, int(epoch)
            self.state = copy.deepcopy(model.state_dict())
        return f1

    def finalize(self, model) -> dict:
        if not self.enabled:
            return {"enabled": False}
        model.load_state_dict(self.state)
        return {"enabled": True, "rule": "max validation macro-F1 per epoch, ties to the earliest",
                "selected_epoch": self.epoch, "best_val_macro_f1": self.best,
                "val_macro_f1_by_epoch": self.curve}


def precision_scores(model, score_fn, y_true, class_names) -> dict:
    """score_fn(model) -> (y_pred, y_score); returns metrics at fp32, per-tensor and per-channel int8.

    The random-number state is saved on entry, restored before each of the three passes and
    restored again on exit. A model that samples at inference (TASNN draws Bernoulli spikes)
    therefore sees, in every captured pass, exactly the draws the pipeline's own test pass
    sees afterwards, and the capture leaves the pipeline's stream untouched. Deterministic
    models are unaffected."""
    import torch
    from src.metrics import compute_metrics, compute_binary_ids_metrics
    from src.models.tree_to_spike import quantize_int8_, quantize_int8_per_channel_
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def restore():
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)

    out = {}
    for name, fn in (("fp32", None), ("per_tensor_int8", quantize_int8_),
                     ("per_channel_int8", quantize_int8_per_channel_)):
        m = model if fn is None else fn(copy.deepcopy(model))
        restore()
        pred, score = score_fn(m)
        out[name] = {"metrics": compute_metrics(y_true, pred, y_score=score, class_names=class_names),
                     "binary_ids_metrics": compute_binary_ids_metrics(y_true, pred, normal_class_index=0)}
    restore()
    return out
