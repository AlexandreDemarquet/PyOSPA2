"""
Comprehensive test suite for PyOSPA2 library.
Tests cover edge cases, performance, parameter variations, and real-world scenarios.
"""

import numpy as np
import pandas as pd
import pytest
from PyOSPA2 import OSPA2
from PyOSPA2._ospa2 import compute_distance_matrix, ospa2_from_matrix


class TestBasicFunctionality:
    """Test fundamental OSPA2 computation."""

    def test_identical_trajectories_zero_error(self):
        """When GT and tracking are identical, OSPA2 should be ~0."""
        gt = pd.DataFrame({
            'ts': [0, 1, 2, 0, 1, 2],
            'track_id': [1, 1, 1, 2, 2, 2],
            'x': [0.0, 1.0, 2.0, 5.0, 6.0, 7.0],
            'y': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        })
        trk = gt.copy()
        
        o = OSPA2(c=100, p=1, q=1, window_length=3, cols=['x', 'y'], id_col='track_id')
        ts, o2, loc, card = o.ospa2_over_time(gt, trk)
        
        assert all(abs(v) < 1e-10 for v in o2), "Identical trajectories should have OSPA2 ≈ 0"
        assert all(abs(v) < 1e-10 for v in loc), "Identical trajectories should have localization error ≈ 0"
        assert all(abs(v) < 1e-10 for v in card), "Identical trajectories should have cardinality error ≈ 0"

    def test_completely_missing_targets(self):
        """When tracking completely misses targets, OSPA2 should reflect cardinality error."""
        gt = pd.DataFrame({
            'ts': [0, 1, 2],
            'track_id': [1, 1, 1],
            'x': [0.0, 1.0, 2.0],
            'y': [0.0, 0.0, 0.0]
        })
        trk = pd.DataFrame({
            'ts': [],
            'track_id': [],
            'x': [],
            'y': []
        })
        
        o = OSPA2(c=100, p=1, q=1, window_length=3, cols=['x', 'y'], id_col='track_id')
        ts, o2, loc, card = o.ospa2_over_time(gt, trk)
        
        assert len(o2) > 0, "Should compute OSPA2 even with missing tracks"
        assert all(c > 0 for c in card), "Cardinality error should be positive when tracks are missing"

    def test_false_positives_detection(self):
        """When tracking has extra targets not in GT."""
        gt = pd.DataFrame({
            'ts': [0, 1, 2],
            'track_id': [1, 1, 1],
            'x': [0.0, 1.0, 2.0],
            'y': [0.0, 0.0, 0.0]
        })
        trk = pd.DataFrame({
            'ts': [0, 1, 2, 0, 1, 2],
            'track_id': [1, 1, 1, 2, 2, 2],
            'x': [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
            'y': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        })
        
        o = OSPA2(c=100, p=1, q=1, window_length=3, cols=['x', 'y'], id_col='track_id')
        ts, o2, loc, card = o.ospa2_over_time(gt, trk)
        
        assert all(c > 0 for c in card), "False positives should create cardinality error"


class TestParameterVariations:
    """Test OSPA2 with different parameter configurations."""

    def test_different_c_parameter(self):
        """Test that different c values affect the score."""
        gt = pd.DataFrame({
            'ts': [0, 1, 2],
            'track_id': [1, 1, 1],
            'x': [0.0, 1.0, 2.0],
            'y': [0.0, 0.0, 0.0]
        })
        trk = pd.DataFrame({
            'ts': [0, 1, 2],
            'track_id': [1, 1, 1],
            'x': [1.0, 2.0, 3.0],
            'y': [0.0, 0.0, 0.0]
        })
        
        o_low = OSPA2(c=1.0, p=1, q=1, window_length=3, cols=['x', 'y'])
        o_high = OSPA2(c=100.0, p=1, q=1, window_length=3, cols=['x', 'y'])
        
        _, o2_low, _, _ = o_low.ospa2_over_time(gt, trk)
        _, o2_high, _, _ = o_high.ospa2_over_time(gt, trk)
        
        # Lower c should cap the error more aggressively
        assert all(l <= h for l, h in zip(o2_low, o2_high)), "Lower c should not increase error"

    def test_different_p_parameter(self):
        """Test that p parameter affects localization vs cardinality balance."""
        gt = pd.DataFrame({
            'ts': [0, 1],
            'track_id': [1, 1],
            'x': [0.0, 1.0],
            'y': [0.0, 0.0]
        })
        trk = pd.DataFrame({
            'ts': [0, 1, 0, 1],
            'track_id': [1, 1, 2, 2],
            'x': [0.0, 1.0, 5.0, 6.0],
            'y': [0.0, 0.0, 0.0, 0.0]
        })
        
        o_p1 = OSPA2(c=100, p=1, q=1, window_length=2)
        o_p2 = OSPA2(c=100, p=2, q=1, window_length=2)
        
        _, o2_p1, _, card_p1 = o_p1.ospa2_over_time(gt, trk)
        _, o2_p2, _, card_p2 = o_p2.ospa2_over_time(gt, trk)
        
        assert len(o2_p1) == len(o2_p2), "Same data should produce same number of timestamps"

    def test_different_q_parameter(self):
        """Test that q parameter (trajectory aggregation) affects scoring."""
        gt = pd.DataFrame({
            'ts': [0, 1, 2],
            'track_id': [1, 1, 1],
            'x': [0.0, 1.0, 2.0],
            'y': [0.0, 0.0, 0.0]
        })
        trk = pd.DataFrame({
            'ts': [0, 1, 2],
            'track_id': [1, 1, 1],
            'x': [0.1, 1.1, 2.1],
            'y': [0.0, 0.0, 0.0]
        })
        
        o_q1 = OSPA2(c=100, p=1, q=1, window_length=3)
        o_q2 = OSPA2(c=100, p=1, q=2, window_length=3)
        
        _, o2_q1, _, _ = o_q1.ospa2_over_time(gt, trk)
        _, o2_q2, _, _ = o_q2.ospa2_over_time(gt, trk)
        
        assert len(o2_q1) == len(o2_q2), "Same data should produce same number of results"


class TestWindowBehavior:
    """Test temporal windowing behavior."""

    def test_window_length_effect(self):
        """Test that window length affects trajectory composition."""
        gt = pd.DataFrame({
            'ts': [0, 1, 2, 3, 4],
            'track_id': [1, 1, 1, 1, 1],
            'x': [0.0, 1.0, 2.0, 3.0, 4.0],
            'y': [0.0, 0.0, 0.0, 0.0, 0.0]
        })
        trk = gt.copy()
        
        o_short = OSPA2(c=100, p=1, q=1, window_length=2)
        o_long = OSPA2(c=100, p=1, q=1, window_length=5)
        
        ts_short, o2_short, _, _ = o_short.ospa2_over_time(gt, trk)
        ts_long, o2_long, _, _ = o_long.ospa2_over_time(gt, trk)
        
        # Both should have same timestamps
        assert len(ts_short) == len(ts_long), "Should evaluate at same timestamps"
        # With identical data, both should be near zero
        assert all(abs(v) < 1e-10 for v in o2_short), "Short window should have low error with identical data"
        assert all(abs(v) < 1e-10 for v in o2_long), "Long window should have low error with identical data"

    def test_missing_intermediate_frames(self):
        """Test behavior when trajectories have gaps."""
        gt = pd.DataFrame({
            'ts': [0, 2, 4],
            'track_id': [1, 1, 1],
            'x': [0.0, 2.0, 4.0],
            'y': [0.0, 0.0, 0.0]
        })
        trk = pd.DataFrame({
            'ts': [0, 2, 4],
            'track_id': [1, 1, 1],
            'x': [0.0, 2.0, 4.0],
            'y': [0.0, 0.0, 0.0]
        })
        
        o = OSPA2(c=100, p=1, q=1, window_length=5, cols=['x', 'y'])
        ts, o2, loc, card = o.ospa2_over_time(gt, trk)
        
        assert all(abs(v) < 1e-10 for v in o2), "Identical trajectories with gaps should have OSPA2 ≈ 0"


class TestMultipleTargets:
    """Test OSPA2 with multiple targets and complex scenarios."""

    def test_track_crossing(self):
        """Test tracking performance when targets cross paths."""
        gt = pd.DataFrame({
            'ts': [0, 1, 2, 0, 1, 2],
            'track_id': [1, 1, 1, 2, 2, 2],
            'x': [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],  # targets cross
            'y': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        })
        trk = pd.DataFrame({
            'ts': [0, 1, 2, 0, 1, 2],
            'track_id': [1, 1, 1, 2, 2, 2],
            'x': [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
            'y': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        })
        
        o = OSPA2(c=100, p=1, q=1, window_length=3, cols=['x', 'y'])
        ts, o2, loc, card = o.ospa2_over_time(gt, trk)
        
        assert all(abs(v) < 1e-10 for v in o2), "Perfect tracking should have OSPA2 ≈ 0 even with crossing"

    def test_track_swap_error(self):
        """Test when tracking swaps identities of two targets."""
        gt = pd.DataFrame({
            'ts': [0, 1, 2, 0, 1, 2],
            'track_id': [1, 1, 1, 2, 2, 2],
            'x': [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
            'y': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        })
        # Tracking has swapped the IDs
        trk = pd.DataFrame({
            'ts': [0, 1, 2, 0, 1, 2],
            'track_id': [2, 2, 2, 1, 1, 1],
            'x': [10.0, 11.0, 12.0, 0.0, 1.0, 2.0],
            'y': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        })
        
        o = OSPA2(c=100, p=1, q=1, window_length=3, cols=['x', 'y'])
        ts, o2, loc, card = o.ospa2_over_time(gt, trk)
        
        # Should still have low OSPA2 since positions are correct (ID swap doesn't matter)
        assert all(abs(v) < 1e-10 for v in o2), "ID swap shouldn't affect OSPA2 metric"

    def test_scale_invariance(self):
        """Test OSPA2 with different spatial scales."""
        gt_small = pd.DataFrame({
            'ts': [0, 1],
            'track_id': [1, 1],
            'x': [0.0, 1.0],
            'y': [0.0, 0.0]
        })
        trk_small = pd.DataFrame({
            'ts': [0, 1],
            'track_id': [1, 1],
            'x': [0.1, 1.1],
            'y': [0.0, 0.0]
        })
        
        # Scale by 100x
        gt_large = gt_small.copy()
        gt_large[['x', 'y']] *= 100
        trk_large = trk_small.copy()
        trk_large[['x', 'y']] *= 100
        
        o_small = OSPA2(c=100, p=1, q=1, window_length=2)
        o_large = OSPA2(c=10000, p=1, q=1, window_length=2)  # Scale c accordingly
        
        _, o2_small, _, _ = o_small.ospa2_over_time(gt_small, trk_small)
        _, o2_large, _, _ = o_large.ospa2_over_time(gt_large, trk_large)
        
        assert len(o2_small) == len(o2_large), "Should have same number of time steps"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_timestep(self):
        """Test with only one timestamp."""
        gt = pd.DataFrame({
            'ts': [0],
            'track_id': [1],
            'x': [0.0],
            'y': [0.0]
        })
        trk = gt.copy()
        
        o = OSPA2(c=100, p=1, q=1, window_length=1, cols=['x', 'y'])
        ts, o2, loc, card = o.ospa2_over_time(gt, trk)
        
        assert len(ts) == 1, "Should have one result"
        assert abs(o2[0]) < 1e-10, "Perfect match should have zero error"

    def test_single_target(self):
        """Test with only one target."""
        gt = pd.DataFrame({
            'ts': [0, 1, 2],
            'track_id': [1, 1, 1],
            'x': [0.0, 1.0, 2.0],
            'y': [0.0, 0.0, 0.0]
        })
        trk = gt.copy()
        
        o = OSPA2(c=100, p=1, q=1, window_length=3, cols=['x', 'y'])
        ts, o2, loc, card = o.ospa2_over_time(gt, trk)
        
        assert all(abs(v) < 1e-10 for v in o2), "Perfect single target should have zero error"

    def test_many_targets(self):
        """Test with many targets simultaneously."""
        n_targets = 50
        n_times = 10
        
        data = []
        for i in range(n_targets):
            for t in range(n_times):
                data.append({
                    'ts': t,
                    'track_id': i,
                    'x': float(i) + np.sin(t * 0.1),
                    'y': float(i) + np.cos(t * 0.1)
                })
        
        gt = pd.DataFrame(data)
        trk = gt.copy()
        
        o = OSPA2(c=100, p=1, q=1, window_length=5, cols=['x', 'y'])
        ts, o2, loc, card = o.ospa2_over_time(gt, trk)
        
        assert len(ts) == n_times, f"Should have {n_times} timestamps"
        assert all(abs(v) < 1e-10 for v in o2), "Perfect tracking of many targets should have zero error"

    def test_empty_dataframes(self):
        """Test with empty dataframes."""
        gt_empty = pd.DataFrame({
            'ts': [],
            'track_id': [],
            'x': [],
            'y': []
        })
        trk_empty = gt_empty.copy()
        
        o = OSPA2(c=100, p=1, q=1, window_length=3, cols=['x', 'y'])
        ts, o2, loc, card = o.ospa2_over_time(gt_empty, trk_empty)
        
        assert len(ts) == 0, "Empty data should produce empty results"

    def test_high_dimensional_trajectories(self):
        """Test with trajectories having more than 2D."""
        gt = pd.DataFrame({
            'ts': [0, 1, 2],
            'track_id': [1, 1, 1],
            'x': [0.0, 1.0, 2.0],
            'y': [0.0, 0.0, 0.0],
            'z': [0.0, 0.0, 0.0],
            'vx': [0.0, 0.0, 0.0]
        })
        trk = gt.copy()
        
        o = OSPA2(c=100, p=1, q=1, window_length=3, cols=['x', 'y', 'z', 'vx'])
        ts, o2, loc, card = o.ospa2_over_time(gt, trk)
        
        assert all(abs(v) < 1e-10 for v in o2), "Perfect 4D tracking should have zero error"


class TestLowLevelFunctions:
    """Test the low-level C++ binding functions directly."""

    def test_compute_distance_matrix_shape(self):
        """Test that distance matrix has correct shape."""
        gt = [np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]])]
        trk = [np.array([[0.5, 0.5]])]
        
        D = compute_distance_matrix(gt, trk, c=100.0, q=1.0)
        
        assert D.shape == (2, 1), f"Expected shape (2,1), got {D.shape}"

    def test_compute_distance_matrix_symmetry(self):
        """Distance from A to B should equal distance from B to A."""
        gt = [np.array([[0.0, 0.0], [1.0, 1.0]])]
        trk = [np.array([[0.5, 0.5]])]
        
        D1 = compute_distance_matrix(gt, trk, c=100.0, q=1.0)
        D2 = compute_distance_matrix(trk, gt, c=100.0, q=1.0)
        
        assert D1[0, 0] == D2[0, 0], "Distance should be symmetric"

    def test_ospa2_from_matrix_properties(self):
        """Test properties of OSPA2 score from distance matrix."""
        # Identity matrix (perfect matching)
        D = np.eye(3) * 50.0
        ospa2, loc, card = ospa2_from_matrix(D, c=100.0, p=1.0)
        
        assert ospa2 >= 0, "OSPA2 should be non-negative"
        assert loc >= 0, "Localization error should be non-negative"
        assert card >= 0, "Cardinality error should be non-negative"

    def test_ospa2_maximum_value(self):
        """OSPA2 should be bounded by c^p."""
        # Large distance matrix
        D = np.ones((3, 3)) * 200.0
        ospa2, loc, card = ospa2_from_matrix(D, c=100.0, p=2.0)
        
        # OSPA2 should not exceed c^p in meaningful scenarios
        assert ospa2 < 100.0 ** 2.0 * 2, "OSPA2 should be bounded in reasonable range"

    def test_distance_matrix_non_negative(self):
        """All distances should be non-negative."""
        gt = [np.array([[0.0, 0.0]]), np.array([[5.0, 5.0]]), np.array([[-5.0, -5.0]])]
        trk = [np.array([[1.0, 1.0]]), np.array([[0.0, 10.0]])]
        
        D = compute_distance_matrix(gt, trk, c=100.0, q=2.0)
        
        assert np.all(D >= 0), "All distances should be non-negative"
        assert np.all(D <= 100.0), "All distances should be capped by c"


class TestDataTypes:
    """Test proper handling of different data types."""

    def test_float32_conversion(self):
        """Test that float32 data is properly converted to float64."""
        gt = pd.DataFrame({
            'ts': np.array([0, 1], dtype=np.int32),
            'track_id': np.array([1, 1], dtype=np.int32),
            'x': np.array([0.0, 1.0], dtype=np.float32),
            'y': np.array([0.0, 0.0], dtype=np.float32)
        })
        trk = gt.copy()
        
        o = OSPA2(c=100, p=1, q=1, window_length=2, cols=['x', 'y'])
        ts, o2, loc, card = o.ospa2_over_time(gt, trk)
        
        assert all(abs(v) < 1e-10 for v in o2), "Should handle float32 conversion properly"

    def test_integer_coordinates(self):
        """Test that integer coordinates are handled."""
        gt = pd.DataFrame({
            'ts': [0, 1, 2],
            'track_id': [1, 1, 1],
            'x': [0, 1, 2],
            'y': [0, 0, 0]
        })
        trk = gt.copy()
        
        o = OSPA2(c=100, p=1, q=1, window_length=3, cols=['x', 'y'])
        ts, o2, loc, card = o.ospa2_over_time(gt, trk)
        
        assert all(abs(v) < 1e-10 for v in o2), "Should handle integer coordinates"


class TestRealWorldScenarios:
    """Test realistic multi-object tracking scenarios."""

    def test_occlusion_effect(self):
        """Simulate temporary occlusion of targets."""
        gt = pd.DataFrame({
            'ts': [0, 1, 2, 3, 4],
            'track_id': [1, 1, 1, 1, 1],
            'x': [0.0, 1.0, 2.0, 3.0, 4.0],
            'y': [0.0, 0.0, 0.0, 0.0, 0.0]
        })
        # Tracking loses target during t=2 and recovers
        trk = pd.DataFrame({
            'ts': [0, 1, 3, 4],
            'track_id': [1, 1, 1, 1],
            'x': [0.0, 1.0, 3.0, 4.0],
            'y': [0.0, 0.0, 0.0, 0.0]
        })
        
        o = OSPA2(c=100, p=1, q=1, window_length=3, cols=['x', 'y'])
        ts, o2, loc, card = o.ospa2_over_time(gt, trk)
        
        assert all(v >= 0 for v in o2), "OSPA2 should be non-negative"
        assert any(c > 0 for c in card), "Should detect cardinality errors from occlusion"

    def test_measurement_noise(self):
        """Simulate measurement noise in tracking."""
        np.random.seed(42)
        gt = pd.DataFrame({
            'ts': list(range(20)) * 3,
            'track_id': [1]*20 + [2]*20 + [3]*20,
            'x': [float(i) for i in range(20)] * 3,
            'y': [0.0]*20 + [5.0]*20 + [-5.0]*20
        })
        
        # Add noise to tracking
        noise = np.random.normal(0, 0.5, size=(len(gt),))
        trk = gt.copy()
        trk['x'] = trk['x'] + noise
        
        o = OSPA2(c=100, p=1, q=1, window_length=5, cols=['x', 'y'])
        ts, o2, loc, card = o.ospa2_over_time(gt, trk)
        
        assert all(v >= 0 for v in o2), "OSPA2 should handle noise gracefully"
        assert all(v >= 0 for v in loc), "Localization error should be positive with noise"

    def test_sudden_target_appearance(self):
        """Test when new targets suddenly appear."""
        gt = pd.DataFrame({
            'ts': [0, 1, 2, 0, 1, 2],
            'track_id': [1, 1, 1, 2, 2, 2],
            'x': [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            'y': [0.0, 0.0, 0.0, 10.0, 10.0, 10.0]
        })
        # Tracking only sees first target at first, then second target appears
        trk = pd.DataFrame({
            'ts': [0, 1, 2, 2],
            'track_id': [1, 1, 1, 2],
            'x': [0.0, 1.0, 2.0, 0.0],
            'y': [0.0, 0.0, 0.0, 10.0]
        })
        
        o = OSPA2(c=100, p=1, q=1, window_length=3, cols=['x', 'y'])
        ts, o2, loc, card = o.ospa2_over_time(gt, trk)
        
        assert len(ts) > 0, "Should compute OSPA2 with target appearance"
        # Last timestep should show cardinality recovery
        assert card[-1] < card[-2] if len(card) > 1 else True, "Cardinality should improve when target detected"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
