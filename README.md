[![iris-flower-dataset-analysis](https://github.com/jim2ola/iris_dataset/actions/workflows/main.yml/badge.svg)](https://github.com/jim2ola/iris_dataset/actions/workflows/main.yml)

## Iris_Dataset_Analysis

This project demonstrates advanced data analysis of the iris dataset using **Python**. We will be doing classification by supervised machine learning algorithm - Logistic Regression and unsupervised machine learning algorithm - Kmeans Clustering.

### Problem Statement

This dataset consists of the physical parameters of three species of flower - Versicolor, Setosa, Virginica. The numeric parameters which the dataset contains are Sepal width, Sepal Length, Petal width and Petal length. In this data we will be predicting the classes of the flowers based on these parameters.

-------

### Data Exploration

We will visualise the data using Tableau, an end-to-end data analytics platform that let us prepare, analyse, combine, and share our insights.

#### Visualization I

The pie chart clearly illustrates that there are 50 data points for each iris species (Versicolor, Setosa, Virginica) and a total of 150 data points.

Using bivariate analysis, we explore the concept of relationship between the 4 features with scatterplots. Filtered by species, we observe that both Petal length and Petal width have higher correlation to Species. We also observe that the Iris-Setosa is clearly distinct while Iris-Versicolor and Iris-Virginica have some intersections.

<img src="/assets/images/piechart_scatterplots.png">

#### Visualization II

The histogram address the same points as above.

<img src="/assets/images/histogram.png">

-------

### Advanced Data Analysis (Python)

In this data exploration, we retrieved crucial information such as skewness, correlation and even feature importances by XGBClassifier to help us made informed choices. In the end, we decided to keep only PetalLengthCm for our prediction.

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

#### Git Clone This Repository
```code
git clone https://github.com/jim2ola/iris_dataset.git
```

#### Install Required Libraries
```code
Make install
```

-------

### Prediction (Supervised)

As mentioned above, we will be using **Logistic Regression** for the demonstration of classifying the iris dataset. We will also be using some common machine learning metrics that will help us gauge the performance of our model. In the end, we achieved a model accuracy of 100% in 0.0156s runtime with only PetalLengthCm.

#### To Run Prediction
```code
cd processors
python predict_supervised.py
```

<img src="/assets/images/prediction1.png">

### Prediction (Unsupervised)

As mentioned above, we will be using **Kmeans Clustering** for the demonstration of classifying the iris dataset. We will also be using some common machine learning metrics that will help us gauge the performance of our model. In the end, we achieved a model accuracy of 100% in 0.109s runtime with only PetalLengthCm.

#### To Run Prediction
```code
cd processors
python predict_unsupervised.py
```

<img src="/assets/images/prediction2.png">
