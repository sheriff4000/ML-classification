from dataset import Dataset

import numpy as np

class Test:
    def __init__(self) -> None:
        data = np.loadtxt(
            "intro2ML-coursework1/wifi_db/noisy_dataset.txt", delimiter=" ")
        set = Dataset(data)
        assert (len(set.unique_labels()[0]) == 4)
        assert (set.unique_labels()[0][0] == 1)
