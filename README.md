# Tweets sentiment classification

## Project overview
This project focuses on classifying the sentiment of tweets using state-of-the-art natural language processing models.
The goal is to accurately determine whether the sentiment expressed in a tweet is positive, negative, neutral, or irrelevant. 
This repository provides notebooks for exploratory data analysis (EDA), model training, and making predictions, as well as utility functions to support these tasks.


## Repository structure

- Notebooks:
    - EDA analysis: [EDA_analysis.ipynb](notebooks/EDA_analysis.ipynb)
    - Model training: [model_training.ipynb](notebooks/model_training.ipynb)
    - Get predictions: [get_predictions.ipynb](notebooks/get_predictions.ipynb)
- Source directory:
     - Utility functions: [src/utils.py](src/utils.py)
     - Model manager:
       - Sentiment class: [src/model_manager/sentiment_dataset.py](src/model_manager/sentiment_dataset.py)


## Installation instructions
To set up the project environment, clone the repository and install the required dependencies from the [requirements.txt](requirements.txt) file:

```shell
pip install -r requirements.txt
```

## Usage guidelines

### Model training and getting predictions
1. Train the BERT model using Jupyter Lab:
   ```shell
   jupyter lab notebooks/get_predictions.ipynb
   ```
2. Get predictions with the trained model:
   ```shell
   jupyter lab notebooks/get_predictions.ipynbb
   ```