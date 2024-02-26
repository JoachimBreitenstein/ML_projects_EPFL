#!/usr/bin/python3 
#author: Joachim Breitenstein 
#date: 31/10/2023

#load packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

############# Load training and testing data from CSV files and pre-processing #############
training_path = "data/training.csv" 
testing_path = "data/testing.csv"    
    
train_data = pd.read_csv(training_path, header = None)
test_data = pd.read_csv(testing_path, header = None)

#Parse features and labels (1,2,3,4)
X_train = train_data.drop(0, axis=1)
y_train = train_data[0]
X_test = test_data.drop(0, axis=1)
y_test = test_data[0]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

##################################################################################################
#Function fitting ordinary logistic regression and testing
def logist_reg():
    # Create model object
    model = LogisticRegression(solver='newton-cg', multi_class='multinomial')

    #train model based on training dataset 
    model.fit(X_train, y_train)

    # Predictions on training set
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.2%}")

    # Predictions on testing set
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Testing Accuracy: {test_accuracy:.2%}")

    # Detailed classification report
    print("\nClassification Report (Testing Set):")
    print(classification_report(y_test, y_test_pred))

    # If you want to see the coefficients
    print(f"Model coefficients: {model.coef_}")

################################################################################################
#necessary functions for running multinomial logistic regression w. gradient ascent and sdg 
def gradient_ascent(X, y, X_test, y_test, learning_rate=0.01, num_iterations=150):
    N, D = X.shape
    K = len(np.unique(y))
    theta = np.random.randn(D, K) #initiate random theta values for first iteration
    
    #initiate lists for saving accuracy values for train + test
    train_accuracies = [] 
    test_accuracies = [] 

    #train + test loop 
    for i in range(num_iterations):
        gradient = compute_gradient(X, y, theta)
        theta += learning_rate * gradient

        #train 
        y_train_pred = prediction_softmax(X, theta)
        train_acc = accuracy_score(y, y_train_pred)  
        train_accuracies.append(train_acc)  
        
        #test 
        y_test_pred = prediction_softmax(X_test, theta)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_accuracies.append(test_acc) 
        
        print(f"Iteration:{i+1}. Training Accuracy: {train_acc:.2%}, Testing Accuracy: {test_acc:.2%}")
        
    # Plotting test + train accuracies
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Testing Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Gradient ascent')
    plt.savefig('accuracies_plot.png', dpi=300)
    plt.show()

    return theta #return optimized theta vals 

def compute_gradient(X, y, theta):
    N, K = X.shape[0], theta.shape[1]
    probs = softmax(X.dot(theta))
    
    # Adjusting for 1-indexed classes
    y_encoded = np.eye(K)[y.astype(int) - 1]
    
    gradient = X.T.dot(y_encoded - probs)
    return gradient

#defining softmax
def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=1, keepdims=True)

#prediction w. softmax
def prediction_softmax(X, theta):
    
    probs = softmax(X.dot(theta))
    # Adjusting for 1-indexed classes
    return np.argmax(probs, axis=1) + 1

def sgd(X, y, X_test, y_test, learning_rate=0.01, num_epochs=50, batch_size=1):
    
    N, D = X.shape
    K = len(np.unique(y))
    theta = np.random.randn(D, K) #initialize random vals for theta

    train_accuracies = []  
    test_accuracies = [] 
    
    for epoch in range(num_epochs):
        # Shuffle dataset before creating batches
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        y_shuffled = y[indices] 
        
        for i in range(0, N, batch_size):
            #creating batches w. samples w. size = batch_size 
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            #compute gradient
            gradient = compute_gradient(X_batch, y_batch, theta)
            theta += learning_rate * gradient
            
            #train + test accuracy
            y_train_pred = prediction_softmax(X, theta)
            y_test_pred = prediction_softmax(X_test, theta)
            train_acc = accuracy_score(y, y_train_pred)  
            train_accuracies.append(train_acc)
            y_test_pred = prediction_softmax(X_test, theta)
            test_acc = accuracy_score(y_test, y_test_pred)
            test_accuracies.append(test_acc)
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}/{N}: Testing Accuracy: {test_acc:.2%}")
            
    # Plotting test + train accuracies
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Testing Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'stochastic gradient ascent, Batch size: {batch_size}')
    plt.savefig(f'accuracies_plot_{batch_size}.png', dpi=300)
    plt.show()
        
    return theta
##################################################################################################

def logist_reg_gradient():
    # Training
    theta_optimal = gradient_ascent(X_train, y_train, X_test, y_test)
    
    # Predictions and metrics
    y_train_pred = prediction_softmax(X_train, theta_optimal)
    y_test_pred = prediction_softmax(X_test, theta_optimal)

    print(f"Final model coefficients: {theta_optimal}")
    print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.2%}")
    print(f"Testing Accuracy: {accuracy_score(y_test, y_test_pred):.2%}")

def logist_reg_sgd():
    # running sdg multinumial logistic regression for batch sizes 1, 16 & 32
    for batch_size in [1, 16, 32]:
        print(f"\nTraining with batch size: {batch_size}")
        #theta_optimal = sgd(X_train, y_train, batch_size=batch_size)
        theta_optimal = sgd(X_train, y_train, X_test, y_test, batch_size=batch_size)

        # Predictions and metrics
        y_train_pred = prediction_softmax(X_train, theta_optimal)
        y_test_pred = prediction_softmax(X_test, theta_optimal)

        print(f"Final model coefficients: {theta_optimal}")
        print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.2%}")
        print(f"Testing Accuracy: {accuracy_score(y_test, y_test_pred):.2%}")

if __name__ == "__main__":
    logist_reg()
    logist_reg_gradient()
    logist_reg_sgd()