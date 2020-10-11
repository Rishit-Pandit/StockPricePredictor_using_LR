# Stock Price Predictor using a Linear Regression
A **Linear Regression** model works on a simple formula and draws a line between the two different sets data points to separate them and predict the next value in the dataset. It uses `Gradient Descent` and `Error Optimisation` to reduce the error and get better and better with its predictions. In this notebook we will use `pandas` to get the datasets and read them and `numpy` to organise the datsets and `sci-kit learn` to prepare our model and predict the closing stock prices of any given company by just looking at the opening price of the stocks and `matplotlib` to visualize the datasets.
## How this works
This model uses **Linear Regression** which works by drawing a line across a 2d plane ploting the `x_train` and `y_train` together with the minimum error. It uses the formula `y = mx + b` - where `y` is the dependent value, `m` is the slope of the line, `x` is the independent variable and `b` is the point from where the line starts - to predict the next y value on the line which it just drew.
## Requirements
To use this model you have to install the following modules:
```bash
pip install pandas
pip install numpy
pip install sklearn
pip install matplotlib
```
Then you can just execute the following command in the terminal - 
```bash
python StockPricePredictor_using_LR.py
```
and use the model that I have prepared.
## Do It Yourself Step-by-Step Guide
Now if you want to build it yourself and understand the exact working of `Linear Regression` the you can just read through this guide.
#### 1. Install the dependencies
We'll need `pandas`, `numpy`, `sklearn` and `matplotlib` for the basic working of our model so after installing them import them using:
```python
import pandas as pd
import numpy as np
import sklearn.linear_model as linear_model
import matplotlib.pyplot as plt
```
#### 2. Getting our dataset
If you have your data in a csv format then youcan simply use the `read_csv` function of the `pandas` module or else there are plenty of different formats of datasets that pandas module can understand and read. So the next line of code would be getting our dataset as `train_dataset` and `test_dataset` from the `train.csv` and `test.csv` files respectively.
Type the following code to do so - 
```python
train_dataset = pd.read_csv("train.csv", sep=",")
test_dataset = pd.read_csv("test.csv", sep=",")
```
Then we'll change the dataset we got into an array using numpy - 
```python
x_train = np.array(train_dataset[["Open"]])
y_train = np.array(train_dataset[["Close"]])
x_test = np.array(test_dataset[["Open"]])
y_test = np.array(test_dataset[["Close"]])
```
#### 3. Training the model
now we'll train our <b>model</b> now using the `x_train`, `y_train` dataset variables and see whether it can correctly predict the closing prices of the company stocks. So type the code - 
```python
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
```
Here we have used `Linear Regression` function of the `sklearn.linear_model` module to create our model.
#### 4. Visualizing the dataset
Now we can visualize the dataset using `matplotlib`. So type the code - 
```python
plt.scatter(x_train, y_train, color="Blue")
plt.show()
```
You will get an output like this :

<img src="data:image/png;base64,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">

#### 5. Making prdictions
Now we can make predictions and predict the closing stock prices with just the opening price of the stocks. Then we will compare the predicted price with the real price. So type the code - 
```python
predictions = linear.predict(x_test)
real = y_test
print("Prediction: ", predictions, " | Real: ", real)
```
You might get the output like this :
```bash
Prediction:  [[290.75005121]]  | Real:  [[285.3]]
```
#### 6. Calculating Error
Now we can calculate the error by subtracting the Predicted value from the real value. So type the code - 
```python
error = predictions - real
print("Error: ", error)
```
You might get the output like this :
```bash
Error:  [[5.45005121]]
```
## Usage 
This model can be used for real time stock price prediction with some more modifications. If you use a much larger and better training dataset to train the model then it will give a more accurate ouput and a less error.
