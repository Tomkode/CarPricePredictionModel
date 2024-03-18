import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
def main():
    selectedcols = ["year", "make", "model", "transmission", "condition", "odometer", 'mmr', "sellingprice"] # the last column is the target
    # Read the CSV file as a pandas DataFrame and drop the rows that contain null values
    dataFrame = pd.read_csv("car_prices.csv")[selectedcols].dropna()
    features = selectedcols[:-1]
    x = dataFrame[features]
    y = dataFrame [ selectedcols[-1] ]
    
    #encode the features
    x = OrdinalEncoder().fit_transform(x)
    y = y.to_numpy().reshape(-1,1)

    #scale the data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(y)
    
    #split the data into train data and test data
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2, random_state=0)
    # use Linear Regression to find the line of best fit for the training data
    model = LinearRegression().fit(X_train,Y_train)
    # get the model's accuracy for the training/test data
    acc = model.score(X_train,Y_train)
    testAcc = model.score(X_test, Y_test)
    print(f"Training Accuracy : {acc:.2f} ") 
    print(f"Test accuracy : {testAcc:.2f}")
    
    
main()