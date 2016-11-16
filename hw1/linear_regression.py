import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#data = pd.read_csv('train.csv',encoding="utf-8")
data = pd.read_csv('train.csv',encoding="latin1")
#data = pd.read_csv('train.csv',encoding="gb18030")
print(data.iloc[0:5])
