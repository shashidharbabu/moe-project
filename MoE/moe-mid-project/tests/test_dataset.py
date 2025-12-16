import unittest
from src.data.moe_mid_dataset import MoEMidDataset

class TestMoEMidDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = MoEMidDataset()

    def test_length(self):
        self.assertGreater(len(self.dataset), 0, "Dataset should not be empty")

    def test_image_loading(self):
        image, label = self.dataset[0]
        self.assertIsNotNone(image, "Loaded image should not be None")
        self.assertIsInstance(image, torch.Tensor, "Loaded image should be a tensor")
        self.assertEqual(label.shape, (3,), "Label should have shape (3,) for domain probabilities")

    def test_normalization(self):
        image, _ = self.dataset[0]
        self.assertTrue((image >= 0).all() and (image <= 1).all(), "Image pixel values should be normalized between 0 and 1")

if __name__ == '__main__':
    unittest.main()