# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:09:18 2021

@author: junyanee
"""

import plotly.express as px

df = px.data.iris()
fig = px.scatter_3d(df, x = 'sepal_length', y = 'sepal_width', z = 'petal_width', color = 'species')
fig.show(renderer = "browser")