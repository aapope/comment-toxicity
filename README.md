
## Setup

##### Get data from the Kaggle site
All data is from [this Kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data). Download and extract the files into the `data/` directory.

##### Install requirements.txt

##### Install nltk dependencies
Run this inside the `setup.ipynb` notebook.


## Run

Start with the `eda.ipynb` notebook. Along with some EDA charts, the main point here is to combine Kaggle's train and test data into a single dataset that will be split later into train/test.

Next is the `prep.ipynb` notebook. This cleans and encodes the text for input into the model.

The use `train.ipynb` to train the model.