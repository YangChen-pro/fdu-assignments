"""手写 MLP 的基础回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlp_hw1.config import build_search_config, build_train_config
from mlp_hw1.data import build_cache_stem
from mlp_hw1.metrics import confusion_matrix
from mlp_hw1.trainer import build_search_candidates, evaluate_split

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - depends on the local environment
    cp = None

if cp is not None:
    from mlp_hw1.backend import to_numpy
    from mlp_hw1.model import ThreeLayerMLP


class ThreeLayerMLPTest(unittest.TestCase):
    """不依赖 EuroSAT 数据集的小规模测试。"""

    @unittest.skipIf(cp is None, "cupy 未安装，跳过 GPU 相关单测")
    def test_forward_shape(self) -> None:
        model = ThreeLayerMLP(input_dim=6, hidden_dim=8, hidden_dim2=6, output_dim=3, activation="relu", xp=cp)
        features = cp.random.randn(4, 6, dtype=cp.float32)
        logits = model.forward(features)
        self.assertEqual(logits.shape, (4, 3))

    @unittest.skipIf(cp is None, "cupy 未安装，跳过 GPU 相关单测")
    def test_training_step_reduces_loss(self) -> None:
        rng = np.random.default_rng(0)
        features = cp.asarray(rng.normal(size=(48, 4)).astype(np.float32))
        labels = cp.asarray((to_numpy(features)[:, 0] + 0.8 * to_numpy(features)[:, 1] > 0).astype(np.int64))
        model = ThreeLayerMLP(
            input_dim=4,
            hidden_dim=16,
            hidden_dim2=8,
            output_dim=2,
            activation="tanh",
            xp=cp,
            seed=0,
            dropout_rate=0.2,
        )
        initial_loss = model.compute_loss(features, labels)
        for _ in range(120):
            model.loss_and_backward(features, labels, weight_decay=0.0)
            model.step(learning_rate=0.05)
        final_loss = model.compute_loss(features, labels)
        self.assertLess(final_loss, initial_loss)

    def test_confusion_matrix_counts(self) -> None:
        matrix = confusion_matrix(
            np.array([0, 0, 1, 2, 2]),
            np.array([0, 1, 1, 2, 0]),
            num_classes=3,
        )
        expected = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]])
        np.testing.assert_array_equal(matrix, expected)

    def test_named_presets_match_report_runs(self) -> None:
        final_p = build_train_config("final_p")
        self.assertEqual(final_p.hidden_dim, 1280)
        self.assertEqual(final_p.hidden_dim2, 768)
        self.assertEqual(final_p.dropout_rate, 0.15)
        self.assertEqual(final_p.epochs, 44)

        final_o = build_train_config("final_o")
        self.assertEqual(final_o.dropout_rate, 0.18)
        self.assertEqual(final_o.epochs, 42)

        final_a = build_train_config("final_a")
        self.assertEqual(final_a.dropout_rate, 0.0)
        self.assertEqual(final_a.epochs, 36)

        final_k = build_train_config("final_k")
        self.assertEqual(final_k.dropout_rate, 0.10)
        self.assertEqual(final_k.epochs, 40)

        final_l = build_train_config("final_l")
        self.assertEqual(final_l.dropout_rate, 0.15)
        self.assertEqual(final_l.epochs, 40)

        final_n = build_train_config("final_n")
        self.assertEqual(final_n.dropout_rate, 0.12)
        self.assertEqual(final_n.epochs, 42)

    def test_full_search_preset_builds(self) -> None:
        search_config = build_search_config("full")
        self.assertEqual(search_config.max_trials, 24)
        self.assertIn(1280, search_config.hidden_dims)
        self.assertEqual(search_config.train_config.hidden_dim, 1024)

    def test_search_candidates_cover_core_hyperparameters(self) -> None:
        candidates = build_search_candidates(build_search_config("full"))
        self.assertEqual(len(candidates), 24)
        self.assertEqual({row["learning_rate"] for row in candidates}, set(build_search_config("full").learning_rates))
        self.assertEqual({row["hidden_dim"] for row in candidates}, set(build_search_config("full").hidden_dims))
        self.assertEqual({row["hidden_dim2"] for row in candidates}, set(build_search_config("full").hidden_dims2))
        self.assertEqual({row["weight_decay"] for row in candidates}, set(build_search_config("full").weight_decays))

    def test_cache_stem_changes_with_split_ratio(self) -> None:
        base = build_cache_stem(seed=42, val_ratio=0.15, test_ratio=0.15, limit_per_class=None)
        changed = build_cache_stem(seed=42, val_ratio=0.10, test_ratio=0.20, limit_per_class=None)
        self.assertNotEqual(base, changed)
        self.assertEqual(base, "eurosat_seed42_val150_test150_full")

    def test_evaluate_split_uses_sample_weighted_loss(self) -> None:
        class DummyXP:
            @staticmethod
            def asarray(value):
                return value

        class DummyModel:
            xp = DummyXP()

            def compute_loss(self, batch_x, _batch_y):
                return 1.0 if batch_x.shape[0] == 2 else 3.0

            def predict(self, batch_x):
                return np.zeros(batch_x.shape[0], dtype=np.int64)

        split_images = np.zeros((3, 2), dtype=np.uint8)
        split_labels = np.zeros(3, dtype=np.int64)
        result = evaluate_split(
            model=DummyModel(),
            split=type("Split", (), {"images": split_images, "labels": split_labels})(),
            mean=np.zeros(2, dtype=np.float32),
            std=np.ones(2, dtype=np.float32),
            batch_size=2,
            return_predictions=False,
        )
        self.assertAlmostEqual(result["loss"], 5.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
