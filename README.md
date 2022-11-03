# DoC Intro to ML Coursework 1 - 2022/23

We have created a classifier that will learn from a dataset and evaluate itself with k-fold cross validation.
The code outputs tree metrics:

- Max depth
- Mean depth
- Accuracy
- The confusion matrix

and class metrics:

- Precision
- Recall
- F1

## Dependencies
IDK if we need this
## Running the code

To run the code from the repo directory:

```bash
$ python ./main.py
```

By default this will run the program and output the evaluation metrics for the clean and noisy datasets found in ./intro2ML-coursework1/wifi_db/ . If you would like to run the program on a new dataset simply put the filepath to your dataset in ./datapath.txt and if needs be change the delimiter in ./main.py line 33 to fit the format of your data.

## Credits

- [James Nock](https://github.com/Jpnock)
- [Dom Justice](https://github.com/DomJustice)
- [Louise Davis](https://github.com/ljd20)
- [Sherif Agbabiaka](https://github.com/sheriff4000)