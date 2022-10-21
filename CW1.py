import numpy as np

def read_file(file: str):
    x = []
    y_labels = []
    with open(file, "r") as infile:
        for i, line in enumerate(infile):
            row = line.split()
            x.append(list(map(float, row[:-1])))
            y_labels.append(row[-1])

        x = np.array(x)
        y = np.array(y)

    return (x, y)
