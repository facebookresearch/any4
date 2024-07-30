import unittest
import numpy as np
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kmeans import build_init, build_sample_weight, kmeans, run_kmeans, initialize_centroids

class TestKMeansFunctions(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(100, 2)  # 100 samples, 1 features
        self.n_clusters = 3

    def test_build_init_random(self):
        n_clusters = 5
        init_type = "random"
        result = build_init(self.X, n_clusters, init_type)
        self.assertEqual(result, "random")

    def test_build_init_manual_random(self):
        n_clusters = 5
        init_type = "manual_random"
        result = build_init(self.X, n_clusters, init_type)
        self.assertEqual(result.shape, (n_clusters, self.X.shape[1]))

    def test_build_init_int(self):
        n_clusters = 5
        init_type = "int"
        result = build_init(self.X, n_clusters, init_type)
        self.assertEqual(result.shape, (n_clusters, self.X.shape[1]))

    def test_build_init_pow(self):
        n_clusters = 5
        init_type = "pow"
        result = build_init(self.X, n_clusters, init_type)
        self.assertEqual(result.shape, (n_clusters, self.X.shape[1]))

    def test_build_init_nf4(self):
        n_clusters = 16
        init_type = "nf4"
        result = build_init(self.X, n_clusters, init_type)
        self.assertEqual(result.shape, (n_clusters, self.X.shape[1]))

    def test_build_init_unsupported(self):
        n_clusters = 5
        init_type = "unsupported"
        with self.assertRaises(ValueError):
            build_init(self.X, n_clusters, init_type)

    def test_build_sample_weight_none(self):
        sample_weight_type = None
        result = build_sample_weight(self.X, sample_weight_type)
        self.assertIsNone(result)

    def test_build_sample_weight_tensor(self):
        sample_weight_type = torch.tensor(np.ones(self.X.shape[0]))
        result = build_sample_weight(self.X, sample_weight_type)
        self.assertTrue(np.array_equal(result, np.ones(self.X.shape[0])))

    def test_build_sample_weight_outlier(self):
        sample_weight_type = "outlier_2_1"
        result = build_sample_weight(self.X, sample_weight_type)
        self.assertEqual(result.shape, (self.X.shape[0],))

    def test_build_sample_weight_gradual(self):
        sample_weight_type = "gradual_2_1"
        result = build_sample_weight(self.X, sample_weight_type)
        self.assertEqual(result.shape, (self.X.shape[0],))

    def test_build_sample_weight_unsupported(self):
        sample_weight_type = "unsupported"
        with self.assertRaises(ValueError):
            build_sample_weight(self.X, sample_weight_type)

    def test_kmeans_basic(self):
        X = np.array([0,0,0, 1,1,1, 2,2,2, 3,3,3]).reshape(-1, 1)
        n_clusters = 4
        Xc, centroids, labels = kmeans(X, n_clusters=n_clusters)
        self.assertEqual(Xc.shape, X.shape)
        self.assertEqual(centroids.shape, (n_clusters, X.shape[1]))
        self.assertEqual(labels.shape, (X.shape[0],))
        self.assertTrue(np.allclose(Xc, X, rtol=0, atol=0))
        self.assertTrue(np.allclose(np.sort(centroids, axis=0), np.array([0,1,2,3]).reshape(-1,1), rtol=0, atol=0))

    def test_initialize_centroids_kmeans_plus_plus(self):
        init = 'k-means++'
        centroids = initialize_centroids(self.X, self.n_clusters, init)
        self.assertEqual(centroids.shape, (self.n_clusters, self.X.shape[1]))

    def test_initialize_centroids_random(self):
        init = 'random'
        centroids = initialize_centroids(self.X, self.n_clusters, init)
        self.assertEqual(centroids.shape, (self.n_clusters, self.X.shape[1]))

    def test_initialize_centroids_callable(self):
        init = lambda X, n_clusters, random_state: X[:n_clusters]
        centroids = initialize_centroids(self.X, self.n_clusters, init)
        self.assertEqual(centroids.shape, (self.n_clusters, self.X.shape[1]))

    def test_initialize_centroids_unsupported(self):
        init = 123  # Unsupported type
        with self.assertRaises(ValueError):
            initialize_centroids(self.X, self.n_clusters, init)

    def test_run_kmeans_basic(self):
        init = 'random'
        centroids = initialize_centroids(self.X, self.n_clusters, init)
        inertia, final_centroids, labels = run_kmeans(self.X, centroids, max_iter=100, tol=1e-4, verbose=0, sample_weight=np.ones(self.X.shape[0]))
        self.assertEqual(final_centroids.shape, (self.n_clusters, self.X.shape[1]))
        self.assertEqual(labels.shape, (self.X.shape[0],))
        self.assertIsInstance(inertia, float)

    def test_run_kmeans_convergence(self):
        init = 'random'
        centroids = initialize_centroids(self.X, self.n_clusters, init)
        inertia, final_centroids, labels = run_kmeans(self.X, centroids, max_iter=1, tol=1e-4, verbose=0, sample_weight=np.ones(self.X.shape[0]))
        self.assertEqual(final_centroids.shape, (self.n_clusters, self.X.shape[1]))
        self.assertEqual(labels.shape, (self.X.shape[0],))
        self.assertIsInstance(inertia, float)

if __name__ == '__main__':
    unittest.main()