import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
from hidden import dataset_path, testTXT


#ClEANING DATA SET
df = pd.read_csv(dataset_path)
print(df.info())
print(df.head())

#read the test manufacture text as a dataframe 
df_test_manufacturer = pd.read_csv(testTXT)
print(df_test_manufacturer.info())

#remove the .jpg from filename of the df so ID can match with manufacture df 
df = df.replace(".jpg","",regex=True)
print(df.head()) 

df = df.join()




