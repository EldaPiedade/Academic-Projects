import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)

def generate_point(num_features, num_rows):
    """ Generate X with an additional column with constant 1."""
    X = []
    for i in range(num_rows):
        x = list(np.random.rand(num_features))
        x.append(1)
        X.append(x)
    coef =np.random.rand(num_features+1)
    return np.array(X) , np.array(coef)


def linear_function(coef, x):
    return np.dot(coef,x)

def binary_classifier_2D(num_features, num_rows):
    """ Associates the Linear function with a classification."""
    x, coef = generate_point(num_features, num_rows)
    _ = list(np.random.rand(num_rows))
    y = []
    for i in range(x.shape[0]):
        if linear_function(coef, x[i]) > _[i]:
            y.append(1)

        if linear_function(coef, x[i]) < _[i]:
            y.append(-1)
    return y ,x ,coef, _

def return_data(num_features, num_rows):
    """ Run this function to return the Synthetic Data and coeficients."""
    y ,x ,coef, _ = binary_classifier(num_features, num_rows)
    data = pd.DataFrame(x)
    data.drop(data.columns[-1],axis =1, inplace = True)
    data['y'] = _
    data['Labels'] = y
    return data, coef


def plot(num_features, num_rows):
    """ Line and scatter plots representing the data and the linear classifier.,
        Return the generated data with defined classes."""
    if num_features == 1:
        y,x,coef,_ =binary_classifier_2D(1, 50)
        data = pd.DataFrame(x)
        data.drop(data.columns[-1],axis =1, inplace = True)
        data['y'] = _
        data['Labels'] = y
        data.columns = ['x','y','Class']
        to_plot = coef[0]*x + coef[1]
        plt.figure(figsize = (10,6))
        sns.scatterplot(
            x= 'x',
            y='y',
            data= data,
            hue='Class',
            legend= "full",
            style="Class"
        )
        plt.title('Artificial Data', fontsize = 20)
        plt.plot(x,to_plot,'-r')
        sns.despine()
    else:
        print("This function is for data with one feature only!")
    return data,coef
