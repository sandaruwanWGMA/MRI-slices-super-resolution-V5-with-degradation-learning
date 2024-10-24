import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


def split_dataset(dataset, train_size=0.7, val_size=0.2, test_size=0.1, random_seed=42):
    # First split to separate out the test dataset
    train_val_size = 1.0 - test_size
    train_val_indices, test_indices = train_test_split(
        np.arange(len(dataset)), test_size=test_size, random_state=random_seed
    )

    val_relative_proportion = val_size / train_val_size
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_relative_proportion, random_state=random_seed
    )

    # Create subsets for training, validation, and testing
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset
