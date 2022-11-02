import numpy as np

class Dataset:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.LABEL_COL_INDEX = -1

    def attributes(self):
        return self.dataset[:, :self.LABEL_COL_INDEX]

    def labels(self):
        return self.dataset[:, self.LABEL_COL_INDEX]

    def unique_labels(self):
        return np.unique(self.labels(), return_counts=True)

    def label_entropy(self) -> float:
        labels = self.labels()
        unique_labels, label_count = self.unique_labels()

        assert (np.sum(label_count) == len(labels))
        entropy = 0
        for i in range(len(unique_labels)):
            prob = float(label_count[i]) / float(len(labels))
            entropy -= prob * np.log2(prob)
        return entropy

    def __len__(self):
        return len(self.dataset)
