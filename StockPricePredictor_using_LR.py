def stockPredictor(train_dataset, test_dataset, epochs):
    # Getting the datsets
    train_dataset = pd.read_csv(train_dataset, sep=",")
    train_dataset = train_dataset[["Open", "Close"]]
    test_dataset = pd.read_csv(test_dataset, sep=",")
    test_dataset = test_dataset[["Open", "Close"]]
    predict = "Close"
    x_train = np.array(train_dataset.drop([predict], 1))
    y_train = np.array(train_dataset[predict])
    x_test = np.array(test_dataset.drop([predict], 1))
    y_test = np.array(test_dataset[predict])
    # Training the model
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train, epochs)
    # Visualize the datsets
    plt.scatter(x_train, y_train, color="Blue")
    plt.show()
    # Make predictions
    predictions = linear.predict(x_test)
    real = y_test
    print("Prediction: ", predictions, " | Real: ", real)
    # Calculate the error
    error = predictions - real
    print("Error: ", error)
    
    
# Calling the function    
# Initializing variables
train_dataset = "train.csv"
test_dataset = "test.csv"
# Calling the function
stockPredictor(train_dataset, test_dataset, epochs = 1000)
