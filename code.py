# 데이터셋 준비
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Kaggle API 세팅
import os

# os.environ을 이용하여 Kaggle API Username, Key 세팅하기
os.environ['KAGGLE_USERNAME'] = 'fastcampuskim'
os.environ['KAGGLE_KEY'] = 'c939a1e37f5ca93b6406a66fc8bb08e5'

df = pd.read_csv('AB_NYC_2019.csv')
df.head()
df['room_type'].value_counts
df.info()
df['availability_365'].hist()
print(df)
