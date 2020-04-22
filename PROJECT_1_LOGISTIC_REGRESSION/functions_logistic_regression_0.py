import time
import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss(w,x,y):
    total = 0
    for i in range(len(y)):
        dot = np.dot(w,x[i])
        total = total + ( (y[i]*np.log(sigmoid(dot))) + ((1.0 - y[i])*np.log(1.0 - sigmoid(dot))) )
    return -total/len(y)




def derivative_loss(w,x,y, number_w):

    total = np.zeros(number_w)

    for i in range(len(y)):
        dot_sig = sigmoid( np.dot(w,x[i]) )
        total += (dot_sig - y[i]) * x[i]
    return total/len(y)

def gradient_descent(w,x,y, α, min_loss, number_w):
    """
    This is the newtons method to find w such that the function L is minimized.
    """
    plot_loss = list()
    start = time.time()

    while loss(w,x,y) > min_loss:
    #for i in range(100):
        n = (time.time() - start)
        if  n > 300:
            break
        else:
            w = w - α*derivative_loss(w,x,y, number_w)
            plot_loss.append(loss(w,x,y))

    return w, plot_loss


def predict(i,w,x):
    dot = np.dot(w,x[i])
    return round(sigmoid(dot))

def margin_counts(W, x, gamma):
    ## Compute probability on each test point
    preds = predict_proba(W,x,size)
    preds = np.array(preds)
    ## Find data points for which prediction is at least gamma away from 0.5
    margin_inds = np.where((preds > (0.5+gamma)) | (preds < (0.5-gamma)))[0]

    return float(len(margin_inds))

def predict_proba(W,x,size):
    proba = list()
    for i in list(range(size)):
        dot = np.dot(w,x[i])
        n = sigmoid(dot)
        proba.append(n)
    return proba


def error(w,x,y):
    """
    This is a function that counts the number of misclassifications, given the optimum weights.
    It returns the average error.
    """
    total = 0
    for i in range(len(y)):
        total += np.sum(round(predict(i,w,x))  != y[i])

    return total/len(y)

def accuracy(w,x,y):
    total = 0
    for i in range(len(y)):
        total+= np.sum(round(predict(i,w,x)) == y[i])
    return total/ len(y)
