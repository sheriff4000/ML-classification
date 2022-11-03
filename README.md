# DoC Intro to ML Coursework 1 - 2022/23

![Clean Tree Visualisation](https://user-images.githubusercontent.com/1413854/199836040-3445f1ac-e499-40cd-9a67-37994a67dd28.png)

We have created a classifier that will learn from a dataset and evaluate itself with 10-fold cross validation.

The code outputs the following tree metrics:

- Max depth
- Mean depth
- Accuracy
- The confusion matrix

It also outputs the following class metrics:

- Precision
- Recall
- F1

Outputs will be produced for several trees with

- No-pruning (using a 90/10/00 split between training, test and validation data)
- Pre-pruning (using a 80/10/10 split, although the 10 for the validation set is not used)
- Post-pruning (using a 80/10/10 split)

The report for this project can be found in [report.pdf](report.pdf)

## Dependencies

It is suggested to setup a Python 3.10 virtual environment to run the code.

To do this, run the following instructions:

```bash
$ cd project_directory
$ python3.10 -m venv venv
$ source venv/bin/activate
(venv) $ pip install --upgrade pip
(venv) $ pip install -r requirements.txt
```

## Running the code

To run the code, change directory to the repository root, source the venv and then run the following:

```bash
(venv) $ python main.py
```

By default this will run the program and output the evaluation metrics for the clean and noisy datasets found in [./intro2ML-coursework1/wifi_db](./intro2ML-coursework1/wifi_db).

## Running the model on a new dataset

Put the filepath to your dataset in [datapath.txt](datapath.txt).

By default, the code assumes tab delimited data, however if your data does not match this you can update the delimiter on line 42 of [main.py](main.py).

Then run with the following:

```bash
(venv) $ python main.py
```

## Credits

- [James Nock](https://github.com/Jpnock)
- [Dom Justice](https://github.com/DomJustice)
- [Louise Davis](https://github.com/ljd20)
- [Sherif Agbabiaka](https://github.com/sheriff4000)
