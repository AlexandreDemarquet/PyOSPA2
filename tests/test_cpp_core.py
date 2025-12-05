from ospa2._ospa2 import compute_distance_matrix, ospa2_from_matrix
import numpy as np

def test_compute_distance_matrix():
    A = [np.array([[0.0, 0.0]])]
    B = [np.array([[1.0, 0.0]])]
    D = compute_distance_matrix(A, B)

    assert D.shape == (1, 1)
    assert abs(D[0,0] - 1.0) < 1e-9

def test_ospa2_from_matrix():
    D = np.array([[3.0]])
    score = ospa2_from_matrix(D, 1.0, 2.0)
    assert isinstance(score, tuple)
