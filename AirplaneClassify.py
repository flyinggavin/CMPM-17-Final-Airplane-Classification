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
df_test_manufacturer = pd.read_csv(testTXT, header=None, names =["plane type"])
print(df_test_manufacturer.info())

#remove the .jpg from filename of the df so ID can match with manufacture df 
#df = df.replace(".jpg","",regex=True)

df_test_manufacturer = df_test_manufacturer.loc[:,"plane type"].str.split(pat = ' ', n=1)
print(" ")

print(df_test_manufacturer)

#combines manufactuer with df 
df = df.join(df_test_manufacturer)
print(df.head()) 



