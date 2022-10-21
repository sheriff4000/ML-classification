import numpy as np

def read_file(file: str):
    x = np.loadtxt(file, usecols=(0,1,2,3,4,5,6))
    y = np.loadtxt(file, usecols=7)

    return (x, y)


print(read_file("intro2ML-coursework1/wifi_db/clean_dataset.txt"))
