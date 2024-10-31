# test_kmeans_clustering.py

import unittest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from kmeans_clustering import load_data, preprocess_data, apply_kmeans  # Import functions from your main module

class TestKMeansClustering(unittest.TestCase):

    def setUp(self):
        """Set up a sample DataFrame for testing."""
        self.sample_data = pd.DataFrame({
            'Age': [25, 30, 35, None],
            'Income': [50000, 60000, None, 80000],
            'Family_Size': [3, 4, 5, 2]
        })

    def test_load_data(self):
        """Test loading data function."""
        # This test can check if the load_data function behaves as expected.
        # You might want to mock the file reading process for unit tests.
        # For now, we'll skip this as it requires external files.
        pass  # Implement your test logic

    def test_preprocess_data(self):
        """Test preprocessing function."""
        processed_data = preprocess_data(self.sample_data)
        self.assertFalse(pd.isnull(processed_data).any().any(), "Data should not contain NaN values after preprocessing")

    def test_apply_kmeans(self):
        """Test K-Means application."""
        scaled_data = StandardScaler().fit_transform(self.sample_data.select_dtypes(include=['float64', 'int64']).fillna(0))
        labels = apply_kmeans(scaled_data, n_clusters=2)
        self.assertEqual(len(labels), len(self.sample_data), "Labels should have the same length as the input data")

if __name__ == "__main__":
    unittest.main()
