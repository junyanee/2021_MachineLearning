# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:19:56 2021

@author: junyanee
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:12:52 2021

@author: junyanee
"""

from sklearn import svm, datasets
import pandas as pd
from sklearn.metrics import accuracy_score

d = datasets.load_iris()
randomData = pd.DataFrame(d.data).sample(20)
new_d = d.target[randomData.index]
RandomDataTest = randomData.sample(frac=0.05)
test = randomData.copy()
test.loc[RandomDataTest.index] = [[6.4,3.2,6.0,2.5]]
s = svm.SVC(gamma = 0.1, C = 10)
s.fit(randomData, new_d)
res = s.predict(test)
print("새로운 20개 샘플의 부류는 :", res)
print("정확도 :", accuracy_score(new_d, res))