import unittest

import numpy as np

from data.data_utils import _select_sar_pusb_positives, create_pu_training_set


class TestPUConstructionInvariants(unittest.TestCase):
    def _assert_case_control_is_permuted_and_synced(self, mode: str) -> None:
        features = np.arange(30, dtype=np.float32).reshape(-1, 1)
        labels = np.array([1] * 12 + [0] * 18, dtype=int)

        out = create_pu_training_set(
            features,
            labels,
            labeled_ratio=0.5,
            selection_strategy="random",
            scenario="case-control",
            case_control_mode=mode,
            random_seed=42,
            return_indices=True,
            return_roles=True,
        )
        out_features, out_labels, out_labeled_mask, source_idx, roles = out

        self.assertEqual(len(out_features), len(out_labels))
        self.assertEqual(len(source_idx), len(out_features))
        self.assertEqual(len(roles), len(out_features))
        np.testing.assert_array_equal(out_features.reshape(-1).astype(int), source_idx)
        np.testing.assert_array_equal(out_labels, labels[source_idx])
        np.testing.assert_array_equal(out_labeled_mask, (roles == "L").astype(int))

        n_labeled = int((roles == "L").sum())
        block_order = np.concatenate(
            [
                np.full(n_labeled, "L", dtype="<U1"),
                np.full(len(roles) - n_labeled, "U", dtype="<U1"),
            ]
        )
        self.assertFalse(np.array_equal(roles, block_order))

    def test_nnpu_full_u_case_control_is_permuted_and_synced(self) -> None:
        self._assert_case_control_is_permuted_and_synced("nnpu_full_u")

    def test_story_equal_n_case_control_is_permuted_and_synced(self) -> None:
        self._assert_case_control_is_permuted_and_synced("story_equal_n")

    def test_sar_pusb_selection_uses_source_mean_max_accept_reject(self) -> None:
        seed = 7
        pos_indices = np.array([0, 1, 2, 3, 4], dtype=int)
        pn_probs = np.array([0.05, 0.20, 0.50, 0.80, 1.00], dtype=float)

        scores = pn_probs[pos_indices] / pn_probs[pos_indices].mean()
        accept_prob = scores / scores.max()
        expected_rng = np.random.default_rng(seed)
        accepted = pos_indices[accept_prob > expected_rng.random(len(pos_indices))]
        expected = expected_rng.permutation(accepted)[:3]

        observed = _select_sar_pusb_positives(
            np.random.default_rng(seed),
            pos_indices,
            pn_probs,
            n_selected=3,
        )

        np.testing.assert_array_equal(observed, expected)


if __name__ == "__main__":
    unittest.main()
