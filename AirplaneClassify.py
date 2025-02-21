import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math

df = pd.read_csv("/Users/gavinwang/Desktop/CMPM17_ML/CMPM-17-Final-Airplane-Classification/test.csv")
print(df.info())
print(df.head())