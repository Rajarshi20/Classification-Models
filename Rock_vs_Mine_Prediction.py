#Predict sonar vs rock
#Workflow -- Collect sonar data -> Data Preprocessing -> Train Test split -> Feed to a machine learning model (Use logistics regression model for binary prediction rock or mine) -> Make prediction

#Importing dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def data_collection_processing():

    sonar_data = pd.read_csv('sonar_data.csv', header = None)
    shape = sonar_data.shape #shape is a property; Return a tuple representing the dimensionality of the DataFrame.
    sonar_data.describe() #method that gives a statistical measure lkike count, mean, standard deviation
    value = sonar_data[60].value_counts() #function that counts the values of the column
    sonar_data.groupby(60).mean() # get the mean value for each column grouped by the target variable 

    #separate the features and target
    features = sonar_data.drop(columns = 60, axis=1)
    target = sonar_data[60]
    #print(features)
    #print(target)
    return sonar_data,features,target

def data_train_test_split(data,features,target):
    X_train, X_test,Y_train, Y_test = train_test_split(features,target, test_size=0.1, stratify=target, random_state=1) 
    #startify based on target would keep equal number of instances of both the classes in the training data
    #random_state controls the randomness of the splitting. It ensures that the data splitting process is reproducible. 
    #When you set a specific value for random_state, you guarantee that the same data points will be included in the training and testing sets every time you run the code.
    # print("X_train shape:", X_train.shape)
    # print("Y_train shape:", Y_train.shape)
    # print("X_test shape:", X_test.shape)
    # print("Y_test shape:", Y_test.shape) 
    #print(X_train)
    #print(Y_train)
    return X_train, Y_train, X_test, Y_test 

def model_training(X_train, Y_train):
    model= LogisticRegression()
    model.fit(X_train, Y_train)
    return model

def model_evaluation(model, X_train, Y_train, X_test, Y_test):
    Y_train_prediction = model.predict(X_train)
    training_accuracy_score = accuracy_score(Y_train_prediction, Y_train)
    print(training_accuracy_score)
    Y_test_prediction = model.predict(X_test)
    testing_accuracy_score = accuracy_score(Y_test_prediction, Y_test)
    print(testing_accuracy_score)

def make_prediction(model):
    input_data = (0.0235,0.0291,0.0749,0.0519,0.0227,0.0834,0.0677,0.2002,0.2876,0.3674,0.2974,0.0837,0.1912,0.5040,0.6352,0.6804,0.7505,0.6595,0.4509,0.2964,0.4019,0.6794,0.8297,1.0000,0.8240,0.7115,0.7726,0.6124,0.4936,0.5648,0.4906,0.1820,0.1811,0.1107,0.4603,0.6650,0.6423,0.2166,0.1951,0.4947,0.4925,0.4041,0.2402,0.1392,0.1779,0.1946,0.1723,0.1522,0.0929,0.0179,0.0242,0.0083,0.0037,0.0095,0.0105,0.0030,0.0132,0.0068,0.0108,0.0090)
    input_data_array = np.asarray(input_data) #will transform the input data in an array of 60 rows, 1 col
    print(input_data_array)
    print(input_data_array.shape)
    #You are allowed to have one "unknown" dimension.
    #Meaning that you do not have to specify an exact number for one of the dimensions in the reshape method.
    #Pass -1 as the value, and NumPy will calculate this number for you.
    input_data_reshaped = input_data_array.reshape(1,-1) #will reshape the array into 1 row and 60 cols
    print(input_data_reshaped)
    print(input_data_reshaped.shape)
    prediction = model.predict(input_data_reshaped)
    print("Predicted value for the input data " + prediction)

    
def main():
    data,X,Y = data_collection_processing()
    X_train, Y_train, X_test, Y_test = data_train_test_split(data,X,Y)
    model = model_training(X_train, Y_train)
    model_evaluation(model, X_train, Y_train, X_test, Y_test)
    make_prediction(model)

if __name__ == "__main__":
    main()