# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:19:52 2021

@author: junyanee
"""

from sklearn import datasets
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

digit = datasets.load_digits()
s = svm.SVC(gamma = 0.001)

accuracies_svm = 0
accuracies_svm_mean = 0;
accuracies_svm_arr = []
for i in range (6):
    accuracies_svm =  cross_val_score(s, digit.data, digit.target, cv = 5+i)
    accuracies_svm_mean = cross_val_score(s, digit.data, digit.target, cv = 5+i).mean()*100
    accuracies_svm_arr.append(accuracies_svm_mean) 
    print(5+i,"-fold cross validation by svm")
    print(accuracies_svm)
    print("AVG of accuracies = %0.3f, STD = %0.3f" %(accuracies_svm.mean()*100, accuracies_svm.std()))
    print("\n")

dt = tree.DecisionTreeClassifier()
dt.fit(digit.data, digit.target)

accuracies_DTs = 0
accuracies_DTs_mean = 0;
accuracies_DTs_arr = []
for i in range (6):
    accuracies_DTs =  cross_val_score(dt, digit.data, digit.target, cv = 5+i)
    accuracies_DTs_mean =  cross_val_score(dt, digit.data, digit.target, cv = 5+i).mean()*100
    accuracies_DTs_arr.append(accuracies_DTs_mean) 
    print(5+i,"-fold cross validation by Tree")
    print(accuracies_DTs)
    print("AVG of accuracies = %0.3f, STD = %0.3f" %(accuracies_DTs.mean()*100, accuracies_DTs.std()))
    print("\n")

x_values = [5,6,7,8,9,10]
y_values_1 = [accuracies_svm_arr[0], accuracies_svm_arr[1], accuracies_svm_arr[2], 
              accuracies_svm_arr[3], accuracies_svm_arr[4], accuracies_svm_arr[5]]
y_values_2 = [accuracies_DTs_arr[0], accuracies_DTs_arr[1], accuracies_DTs_arr[2], 
              accuracies_DTs_arr[3], accuracies_DTs_arr[4], accuracies_DTs_arr[5]]
plt.plot(x_values, y_values_1, marker = 'o')
plt.plot(x_values, y_values_2, marker = 's')
plt.title('Comparing SVM to DTs')
plt.xlabel('num of folds')
plt.ylabel('mean of accuracies')
plt.legend(['SVM', 'DTs'])
plt.grid(True)
plt.savefig('result.png')
plt.show()
