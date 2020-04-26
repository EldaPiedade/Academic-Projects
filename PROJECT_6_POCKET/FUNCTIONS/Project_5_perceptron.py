import numpy as np
import pandas as pd

def h(w,x):
    """Perceptron Hypothesis Fuction or Sign Function.
       w = weights.
       x = input for one train example."""
    x_new = list(x)
    x_new.append(1)
    if np.dot(w,np.array(x_new)) > 0:
        return 1
    else:
        return -1

def PLA(w,x,y):
    """Perceptron Learning Algorithm updates weights given a condition."""
    if h(w,x) != y :
        x = list(x)
        x.append(1)
        w = w + (y *  np.array(x) )
    return w

def predictor(w,x,n):
    """ Class Prediction for the nth entry."""
    return h(w,x[n])

def error(w,x,y):
    err = 0
    for i in range(len(y)):
        if h(w,x[i]) != y[i]:
            err += 1
    return err /len(y)

def train_model(w,x,y,epochs):
    # Iterate PLA
    for i in range(epochs):
    # Choose random entries to update
        j = np.random.choice(len(y))
        w = PLA(w,x[j],y[j])
    return w



def pocket(T,x,y,nw,epochs):
    w = np.random.randn(nw)
    w_hat = np.random.randn(nw)

    for i in range(T):
        w = train_model(w,x,y,epochs)
        
        if error(w,x,y) < error(w_hat,x,y):
            w_hat = w.copy()
      
    return w_hat
