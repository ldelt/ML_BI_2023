import numpy as np


def binary_classification_metrics(y_pred, y_true):
    
    y_pred = y_pred.astype(int)
    y_true = y_true.astype(int)
    
    dic = {'TP':0, 'TN':0, 'FP':0, 'FN':0}
    
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            dic['TP'] += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            dic['TN'] += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            dic['FP'] += 1
        else:
            dic['FN'] += 1
    
    if dic['TP'] + dic['FP'] == 0:
        precision = 0
    else:
        precision = dic['TP'] / (dic['TP'] + dic['FP'])
    
    if dic['TP'] + dic['FN'] == 0:
        recall = 0
    else:
        recall = dic['TP'] / (dic['TP'] + dic['FN'])
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    accuracy = (dic['TP'] + dic['TN']) / (dic['TP'] + dic['TN'] + dic['FP'] + dic['FN'])
    
    return dict(precision=precision, recall=recall, f1=f1, accuracy=accuracy)



def multiclass_accuracy(y_pred, y_true):
    
    y_pred = y_pred.astype(int)
    y_true = y_true.astype(int)
    
    a = y_pred == y_true
    total = len(a)
    tr = len(np.where(a == True)[0])
    ac = tr / total
    return ac
    


def r_squared(y_pred, y_true):
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    r2 = 1 - (np.sum(np.power(y_true - y_pred, 2)))/(np.sum(np.power(y_true - y_true.mean(), 2)))
    return r2


def mse(y_pred, y_true):
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    n = len(y_true)
    
    ms = (1/n)*np.sum(np.power(y_true - y_pred, 2))
    
    return ms
    


def mae(y_pred, y_true):
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    n = len(y_true)
    
    ma = (1/n)*np.sum(np.abs(y_true - y_pred))
    
    return ma
