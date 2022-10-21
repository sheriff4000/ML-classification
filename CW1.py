import numpy as np


x = []
y_labels = []
with open("intro2ML-coursework1/wifi_db/clean_dataset.txt", "r") as infile:
    for i, line in enumerate(infile):
        row = line.split()
        x.append(list(map(float, row[:-1])))
        y_labels.append(row[-1])

    x = np.array(x)
    y = np.array(y)
