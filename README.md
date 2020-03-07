# Stock Price Predictor using a Support Vector Machine
A <b>Support Vector Machine (SVM)</b> works on a simple formula and draws a line between the two different sets data points to separate them and predict the next value in the dataset. It uses `Gradient Descent` and `Error Optimisation` to reduce the error and get better and better with its predictions. In this notebook we will use `pandas` to get the datasets and read them and `numpy` to organise the datsets and `sci-kit learn` to prepare our model and predict the closing stock prices of any given company by just looking at the opening price of the stocks and `matplotlib` to visualize the datasets.
## How this works
This model uses a <b>SVM</b> which works by drawing a line across a 2d plane ploting the `x_train` and `y_train` together with the minimum error. It uses the formula `y = mx + b` - where `y` is the dependent value, `m` is the slope of the line, `x` is the independent variable and `b` is the point from where the line starts - to predict the next y value on the line which it just drew.
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
python StockPricePredictor_using_SVM.py
```
and use the model that I have prepared.
## Do It Yourself Step-by-Step Guide
Now if you want to build it yourself and understand the exact working of `Support Vector Machines` the you can just read through this guide.
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
