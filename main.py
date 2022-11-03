from numpy.random import default_rng
import numpy as np

from CW1 import Model
import unit_test

unit_test.Test()

seed = 1030
rng = default_rng(seed)

print("CLEAN DATA")
clean_data = np.loadtxt(
    "intro2ML-coursework1/wifi_db/clean_dataset.txt", delimiter="\t")
clean_model = Model(clean_data, rng)
clean_metrics = clean_model.run()
clean_metrics.print()


print("\n\nNOISY DATA")
noisy_data = np.loadtxt(
    "intro2ML-coursework1/wifi_db/noisy_dataset.txt", delimiter=" ")
noisy_model = Model(noisy_data, rng)
noisy_metrics = noisy_model.run()
noisy_metrics.print()

text_file = open("datapath.txt", "r")
datapath = text_file.read()

if datapath != "":
    print("\n\nUSER DATA")  
    user_data = np.loadtxt(
        datapath, delimiter="\t")
    user_model = Model(user_data, rng)
    user_metrics = user_model.run()
    user_metrics.print()
