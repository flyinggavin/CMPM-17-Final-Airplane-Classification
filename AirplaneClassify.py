import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
from PIL import Image
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


#removing the image/plane id from plane type colum
df['plane type'] = df["plane type"].apply(lambda x:x[1:])
df['plane type'] = df["plane type"].apply(lambda x:"".join(x))
print(df['plane type'])

df = df.drop(columns=["Classes", "Labels"]) #remove unnecessary coloums 


print(df.head())

#########DISPLAY IMAGES############
# for i in range(1, 26):
#     plt.subplot(5, 5, i)
#     plt.imshow(Image.open(f"/Users/gavinwang/Desktop/CMPM17_ML/CMPM-17-Final-Airplane-Classification/Final Project Data/archive/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images/{df.loc[i, "filename"]}"))
    
#     plt.axis("off")
#     plt.title(df.loc[i, "plane type"])
# plt.tight_layout()
# plt.show()

