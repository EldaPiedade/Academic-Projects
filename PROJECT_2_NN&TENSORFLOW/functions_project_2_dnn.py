import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import shutil as sh
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def baseline_model(input_dim,l1,hidden_layers,num_hidden_layer):
    """l1: number of neurons on first layer,
    hidden_layers: an array that contains number of neurons for each hidden layer,
    num_hidden_layer: an integer for the number of hidden layers
    """
    model = tf.keras.Sequential([
        keras.layers.Dense(l1, input_dim=input_dim, activation='relu')])
    for i,j in zip(range(num_hidden_layer),hidden_layers):
        model.add(keras.layers.Dense(j, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])
    return model


def process_data(features,target,i,data):
    X_train_idx = pd.read_csv(f"train_idx{i}.csv", header = None)
    X_train_idx = X_train_idx.values.T[0]
    new = data.loc[X_train_idx][features + [target]].dropna(axis = 0)
    X = new[features]
    Y = new[target]

    X_test_idx = pd.read_csv(f"test_idx{i}.csv", header = None)
    X_test_idx = X_test_idx.values.T[0]
    new_2 = data.loc[X_test_idx][features + [target]].dropna(axis = 0)
    X_valid = new_2[features]
    Y_valid = new_2[target]
    return X,Y,X_valid,Y_valid

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') < 0.20):
            print("\nReached 20% loss so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

def train(input_dim,l1_array,hidden_layers_array,
          num_hidden_layer_array,epochs,model_num_1,
          model_num_2,features,target,data,designation):
    """
    l1_array: an array with # of neurons for the first layer of each model,
    hidden_layers_array:(array of arrays)contains neurons # for each hidden layer per model,
    num_hidden_layer_array: array of integers,
    model_num: an integer indicating # of models to train
    """
    for i,j,k,z in zip(l1_array,hidden_layers_array,
                       num_hidden_layer_array,range(model_num_1,model_num_2)):
        model = baseline_model(input_dim,i,j,k)
        print(model.summary())
        os.makedirs(f'{designation}/Model_{z}')
        for i in range(5):
            X,Y,x_valid,y_valid  = process_data(features,target,i,data)
            history = model.fit(X,Y, epochs =epochs,
                            verbose = True,
                            validation_data = (x_valid,y_valid),
                            callbacks=[callbacks])

            loss = np.asarray(history.history["loss"])
            loss_val = np.asarray(history.history["val_loss"])
            np.savetxt(f'loss{i}.csv', loss, delimiter=',')
            np.savetxt(f'loss_val{i}.csv',loss_val,delimiter=',')

            model.save(f'{designation}/Model_{z}/kfold_{i}')
            sh.move(f'loss{i}.csv',f'{designation}/Model_{z}')
            sh.move(f'loss_val{i}.csv',f'{designation}/Model_{z}')
model_name_list = ['Model_0',
                  'Model_1',
                  'Model_2',
                  'Model_3']


def new_evaluate(model_name,data,features,target):
    """model_name: model name. Example : Model_0
       i: is the fold number, must match for trainidx and testidx
       test_index_path: is the path for indeces of test data for each fold
       data_path: must be a new path to augmented data
       features : 0 to 8 ( 9 total)
       target : column named 9

       Run this function for each data category, and a particular model
       """
    sensit = list()
    specif = list()
    auc = list()
    false_pr = list()
    true_pr = list()
    thres = list()
    acc = list()

    for i in range(5):
        new_model = tf.keras.models.load_model(f"Train_Models/{model_name}/kfold_{i}")
        X_test_idx = pd.read_csv(f'test_idx{i}.csv', header = None)
        X_test_idx = X_test_idx.values.T[0]
        new_2 = data.loc[X_test_idx][features + [target]].dropna(axis = 0)
        X_valid = new_2[features]
        _valid = new_2[target]
        new_y = new_model.predict(X_valid).ravel()
        y_pred_class = np.array([round(x) for x in new_y])
        confusion = metrics.confusion_matrix(_valid, y_pred_class)
        TN, FP, FN, TP = confusion.ravel()
        sensitivity = TP / float(FN + TP)
        specificity = TN / (TN + FP)
        fpr,tpr,thresholds = metrics.roc_curve(_valid,new_y)
        auc_keras = metrics.auc(fpr,tpr)
        accu = metrics.accuracy_score(_valid, y_pred_class)

        sensit.append(sensitivity)
        specif.append(specificity)
        auc.append(auc_keras)
        false_pr.append(fpr)
        true_pr.append(tpr)
        thres.append(thresholds)
        acc.append(accu)

    return  sensit, specif, auc, false_pr, true_pr, thres , acc
