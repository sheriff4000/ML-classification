from numpy.random import default_rng, Generator
import numpy as np

from learning import Model
import unit_test


RNG_SEED = 1030


# Trains the model named `name` on the provided `datapath` data, which is
# separated by `delimiter` and uses a random number generator `rng` to control
# randomness.
def run_model(name: str, datapath: str, delimiter: str, rng: Generator):
    """Trains a new model on the provided dataset and prints evaluation statistics.

    Args:
        name (str): The name of the model
        datapath (str): The path to the dataset to train and evaluate on.
        delimiter (str): The delimiter which separates attributes in the dataset file.
        rng (Generator): A random number generator to produce the same results on each run.
    """
    print(f"Training on {name}...")
    dataset = np.loadtxt(datapath, delimiter=delimiter)
    model = Model(dataset, rng)
    metrics = model.run()
    metrics.print()
    print("\n")


# Run unit tests to make sure core functionality works
unit_test.Test()

# Run the test data sets
rng = default_rng(RNG_SEED)
run_model("Clean Data", "intro2ML-coursework1/wifi_db/clean_dataset.txt", "\t", rng)
run_model("Noisy Data", "intro2ML-coursework1/wifi_db/noisy_dataset.txt", " ", rng)

# Run custom model data
user_datapath = open("datapath.txt", "r").readline().strip()
if user_datapath != "":
    run_model("User Data", user_datapath, "\t", rng)
