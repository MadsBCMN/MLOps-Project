import os
import sys

_TEST_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)
sys.path.append(_PROJECT_ROOT)

from src.data.dataloader import dataloader

def test_data():
    dataset, _ = dataloader()
    print(len(dataset))

test_data()