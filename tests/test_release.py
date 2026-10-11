"""Release checks: the core components run on synthetic data, the claims the README makes
about them hold, the dataset registry is complete, and the record of reported numbers points
at files that exist. CPU only; a few seconds.

    python -m unittest tests.test_release
"""
import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

N_FEATURES, N_CLASSES, T = 6, 3, 20


def toy_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, N_FEATURES)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.8).astype(int) + (X[:, 2] > 0.7).astype(int)
    return X, y


class TestTreeToSpike(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import xgboost as xgb
        from src.models.tree_to_spike import TreeGuidedSpikeEncoder
        cls.X, cls.y = toy_data()
        cls.teacher = xgb.XGBClassifier(n_estimators=20, max_depth=3, n_jobs=1, random_state=0)
        cls.teacher.fit(cls.X, cls.y)
        cls.enc = TreeGuidedSpikeEncoder(T=T).fit(cls.teacher, N_FEATURES, X=cls.X)

    def test_encoder_is_a_time_to_first_spike_code(self):
        s = self.enc.encode(self.X[:32])
        self.assertGreater(self.enc.out_dim, 0)
        self.assertEqual(tuple(s.shape), (T, 32, self.enc.out_dim))
        self.assertTrue(torch.all((s == 0) | (s == 1)))
        self.assertTrue(torch.all(s.sum(0) <= 1), "a channel spikes at most once in the window")

    def test_encoder_edges_are_teacher_split_thresholds(self):
        dump = "".join(self.teacher.get_booster().get_dump())
        thresholds = {float(t) for t in re.findall(r"<([-0-9.e]+)\]", dump)}
        for j, edges in self.enc.edges_.items():
            for e in edges:
                self.assertTrue(any(abs(e - t) < 1e-6 for t in thresholds), f"feature {j}: edge {e}")

    def test_student_runs_and_dense_twin_has_the_same_parameters(self):
        from src.models.tree_to_spike import SNNStudent
        from src.models.dnn_student import DNNStudent
        snn = SNNStudent(self.enc.out_dim, N_CLASSES)
        dnn = DNNStudent(self.enc.out_dim, N_CLASSES)
        logits, *_ = snn(self.enc.encode(self.X[:16]))
        self.assertEqual(tuple(logits.shape), (16, N_CLASSES))
        self.assertEqual(tuple(dnn(self.enc.encode_dense(self.X[:16])).shape), (16, N_CLASSES))
        count = lambda m: sum(p.numel() for p in m.parameters())
        self.assertEqual(count(snn), count(dnn), "DNN-Tree is the parameter-identical twin")


class TestBaselines(unittest.TestCase):
    def test_vanilla_snn_and_tasnn_run(self):
        from src.encoding import get_encoder
        from src.models.snn import SNN
        from src.models.tasnn import TASNN
        X, _ = toy_data(n=8)
        x = torch.from_numpy(X)
        out = SNN(num_features=N_FEATURES, hidden_size=16, num_classes=N_CLASSES,
                  neuron_name="LeakyParallel", num_steps=10)(get_encoder("latency")(x, 10))
        self.assertTrue(torch.isfinite(out[0]).all())
        tasnn = TASNN(num_features=N_FEATURES, num_classes=N_CLASSES, num_steps=8, d_model=16, n_heads=2,
                      n_layers=1, ff=32, stc_channels=8, lif_hidden=(16, 8))
        self.assertEqual(tuple(TASNN.ensemble_logits(tasnn(x)).shape), (8, N_CLASSES))


class TestRegistryAndRecords(unittest.TestCase):
    def test_all_eleven_datasets_are_registered(self):
        from src.data import DATASET_LOADERS
        for name in ("ciciot2023", "ciciomt2024", "edgeiiot", "toniot", "cse-cic-ids2018", "cicids2017",
                     "ctu13", "kddcup99", "nslkdd", "unswnb15", "iscxids2012"):
            self.assertIn(name, DATASET_LOADERS)

    def test_reported_numbers_point_at_existing_files(self):
        prov = json.loads((REPO / "results/tables/numbers_provenance.json").read_text())
        self.assertEqual(len(prov["macros"]), 1205)
        self.assertFalse(prov["dirty"], "the published numbers were generated from a clean tree")
        missing = set()
        for name, m in prov["macros"].items():
            self.assertNotEqual(m["value"], "", name)
            for path in re.findall(r"(?:results|scripts|src|Figures)/[A-Za-z0-9_./*{}<>-]+", m["source"]):
                path = path.rstrip(".,;)")
                if any(c in path for c in "*{}<>"):     # a glob or template, not one file
                    continue
                if not (REPO / path).exists():
                    missing.add(path)
        self.assertEqual(missing, set())


if __name__ == "__main__":
    unittest.main()
