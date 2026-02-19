import sys
from unittest.mock import MagicMock

# Mock dependencies before importing the module under test
sys.modules["numpy"] = MagicMock()
tf_mock = MagicMock()
sys.modules["tensorflow"] = tf_mock

import pytest
from data_preprocessing import split_train_val

def test_split_train_val_invalid_val_split():
    items = list(range(10))
    with pytest.raises(ValueError, match="val_split must be between 0 and 1"):
        split_train_val(items, val_split=0.0)
    with pytest.raises(ValueError, match="val_split must be between 0 and 1"):
        split_train_val(items, val_split=1.0)
    with pytest.raises(ValueError, match="val_split must be between 0 and 1"):
        split_train_val(items, val_split=-0.1)
    with pytest.raises(ValueError, match="val_split must be between 0 and 1"):
        split_train_val(items, val_split=1.1)

def test_split_train_val_insufficient_items():
    with pytest.raises(ValueError, match="Need at least 2 samples"):
        split_train_val([], val_split=0.2)
    with pytest.raises(ValueError, match="Need at least 2 samples"):
        split_train_val([1], val_split=0.2)

def test_split_train_val_min_valid_items():
    items = [1, 2]
    train, val = split_train_val(items, val_split=0.5)
    assert len(train) == 1
    assert len(val) == 1
    assert set(train + val) == set(items)

def test_split_train_val_disjoint_and_complete():
    items = list(range(100))
    train, val = split_train_val(items, val_split=0.2)

    # Check completeness
    assert len(train) + len(val) == len(items)
    assert set(train + val) == set(items)

    # Check disjointness
    assert set(train).isdisjoint(set(val))

def test_split_train_val_determinism():
    items = list(range(50))
    train1, val1 = split_train_val(items, val_split=0.2, seed=42)
    train2, val2 = split_train_val(items, val_split=0.2, seed=42)

    assert train1 == train2
    assert val1 == val2

def test_split_train_val_randomness():
    items = list(range(50))
    # Using different seeds should highly likely produce different splits
    train1, val1 = split_train_val(items, val_split=0.2, seed=42)
    train2, val2 = split_train_val(items, val_split=0.2, seed=43)

    assert train1 != train2 or val1 != val2

def test_split_train_val_proportions():
    items = list(range(100))

    # val_split = 0.2 -> 20 val, 80 train
    train, val = split_train_val(items, val_split=0.2)
    assert len(val) == 20
    assert len(train) == 80

    # val_split = 0.1 -> 10 val, 90 train
    train, val = split_train_val(items, val_split=0.1)
    assert len(val) == 10
    assert len(train) == 90

    # val_split = 0.25 -> 25 val, 75 train
    train, val = split_train_val(items, val_split=0.25)
    assert len(val) == 25
    assert len(train) == 75

def test_split_train_val_edge_proportions():
    items = list(range(10))

    # Very small val_split should still result in at least 1 val sample (because of max(1, ...))
    # int(10 * 0.01) = 0
    # max(1, 0) = 1
    train, val = split_train_val(items, val_split=0.01)
    assert len(val) == 1
    assert len(train) == 9

    # Very large val_split should still result in at least 1 train sample (because of min(..., len-1))
    # int(10 * 0.99) = 9
    # min(9, 9) = 9
    train, val = split_train_val(items, val_split=0.99)
    assert len(val) == 9
    assert len(train) == 1
