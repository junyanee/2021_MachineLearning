# -*- coding: utf-8 -*-
"""
Created on Thu May 27 22:40:12 2021

@author: junyanee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 코인데스크 사이트에서 다운로드한 1년치 비트코인 가격 데이터 읽기
f = open('BTC_USD_2020-05-28_2021-05-27-CoinDesk.csv', 'r')
coindesk_data = pd.read_csv(f, header = 0)
seq = coindesk_data[['Closing Price (USD)']].to_numpy() #종가만 취함
print('데이터 길이:', len(seq),'\n앞쪽 5개 값:', seq[0:5])

# 그래프로 데이터 확인
plt.plot(seq, color = 'red')
plt.title('Bitcoin Prices (1 year from 2019-02-28)')
plt.xlabel('Days'); plt.ylabel('Price in USD')
plt.show()

# 시계열 데이터를 위도우 단위로 자르는 함수
def seq2dataset(seq, window, horizon):
    X = []; Y = []
    for i in range(len(seq) - (window + horizon) + 1):
        x = seq[i : (i + window)]
        y = (seq[i + window + horizon - 1])
        X.append(x); Y.append(y);
        return np.array(X), np.array(Y)
        
w = 7 #윈도우 크기
h = 1 #수평선 계수

X, Y = seq2dataset(seq, w, h)
print(X.shape, Y.shape)
print(X[0], Y[0]); print(X[-1], Y[-1])