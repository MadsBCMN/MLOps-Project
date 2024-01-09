import os
import sys
from torch.utils.data import DataLoader
import tests
print("YOYOYO", tests._PROJECT_ROOT)

from src.data.dataloader import dataloader

def test_data():
    dataset_train, dataset_test = dataloader()
    assert len(dataset_train) == 5712 , "The number of training tensors is not equal to the number of training images"
    assert len(dataset_test) == 1311 , "The number of test tensors is not equal to the number of test images"
    for data, label in dataset_test:
        assert(data.size() == (1,86,86)) , f"The shape of the test tensors are not correct, expected (1,86,86) but recieved {data.size()}"
        assert(label in [0,1,2,3]) , "The labels for the test tensors are not correct, expected labels [0,1,2,3]"
    for data, label in dataset_train:
        assert(data.size() == (1,86,86)) , f"The shape of the train tensors are not correct, expected (1,86,86) but recieved {data.size()}"
        assert(label in [0,1,2,3]) , "The labels for the trains tensors are not correct, expected labels [0,1,2,3]"
