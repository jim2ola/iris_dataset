[![iris-flower-dataset-analysis](https://github.com/jim2ola/iris_dataset/actions/workflows/main.yml/badge.svg)](https://github.com/jim2ola/iris_dataset/actions/workflows/main.yml)

## Iris_Dataset

This project demonstrates data analysis of the iris dataset using **Python**. We will be doing classification by supervised machine learning algorithm - Logistic Regression.

#### Problem Statement

This dataset consists of the physical parameters of three species of flower - Versicolor, Setosa, Virginica. The numeric parameters which the dataset contains are Sepal width, Sepal Length, Petal width and Petal length. In this data we will be predicting the classes of the flowers based on these parameters.

-------

### Advanced Data Analysis (Python)

In this data exploration, we retrieved crucial information such as skewness, correlation and even feature importances by XGBoostRegressor to help us made informed choices. In the end, we decided to keep only PetalLengthCm and PetalWidthCm for our prediction.

<img src="/assets/images/analysis.png">

#### To Run Analysis
```code
cd processors
python analysis.py
```

-------

### Environment Setup

As a good practice for any python projects, we will create a virtual environment to exercise full control via a stable, reproducible, and portable environment.

#### Create A Python Virtual Environment
```code
python3 -m venv ~/.iris-flower-data-analytics
```

#### Activate The Python Virtual Environment
```code
source ~/.iris-flower-data-analytics/bin/activate
```

#### Git Clone this repository
```code
git clone https://github.com/jim2ola/iris_dataset.git
```

#### Install Required Libraries
```code
Make install
```

-------

### Prediction

As mentioned above, we will be using **Logistic Regression** for the demonstration of classifying the iris dataset. We will also be using some common machine learning metrics that will help us gauge the performance of our model. In the end, we achieved a model accuracy of 100% in 0.0156s runtime.

#### To Run Prediction
```code
cd processors
python predict.py
```

<img src="/assets/images/predict.png">