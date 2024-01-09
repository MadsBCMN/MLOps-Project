import os
import sys
from torch.utils.data import DataLoader
sys.path.append(os.path.normcase(os.getcwd()))
from src.data.dataloader import dataloader
from tests.config import assert_images
from src.data.config import image_size

def test_data():
    n_train, lt_train = assert_images("data/raw/Training")
    n_test, lt_test = assert_images("data/raw/Testing")

    dataset_train, dataset_test = dataloader()

    assert len(dataset_train) == n_train, "The number of training tensors is not equal to the number of training images"
    assert len(dataset_test) == n_test, "The number of test tensors is not equal to the number of test images"
    
    l_train = []
    for data, label in dataset_train:
        assert data.size() == (1,) + image_size, f"The shape of the train tensors is not correct, expected (1, 86, 86) but received {data.size()}"
        l_train.append(label.item())
    assert set(l_train) == set(lt_train), f"The labels for the train tensors are not correct, expected labels {lt_train}"

    l_test = []
    for data, label in dataset_test:
        assert data.size() == (1,) + image_size, f"The shape of the test tensors is not correct, expected (1, 86, 86) but received {data.size()}"
        l_test.append(label.item())
    assert set(l_test) == set(lt_test), f"The labels for the test tensors are not correct, expected labels {lt_test}"

test_data()