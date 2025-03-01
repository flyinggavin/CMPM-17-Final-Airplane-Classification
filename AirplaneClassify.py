import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
from PIL import Image
from hidden import dataset_path, testTXT, dataset_path_train, trainTXT

##########TEST DATA##################
#ClEANING DATA SET
df = pd.read_csv(dataset_path) #test data 

#read the test manufacture text as a dataframe 
df_test_manufacturer = pd.read_csv(testTXT, header=None, names =["plane type"])

#remove the .jpg from filename of the df so ID can match with manufacture df 
#df = df.replace(".jpg","",regex=True)
df_test_manufacturer = df_test_manufacturer.loc[:,"plane type"].str.split(pat = ' ', n=1) #split space since txt uses sapce

#combines manufactuer with df 
df = df.join(df_test_manufacturer)


#removing the image/plane id from plane type colum
df['plane type'] = df["plane type"].apply(lambda x:x[1:])
df['plane type'] = df["plane type"].apply(lambda x:"".join(x))

df = df.drop(columns=["Classes", "Labels"]) #remove unnecessary coloums 


print(df.head())

##########TRAIN DATA##################
dftrain = pd.read_csv(dataset_path_train) #trian dataset
df_train_manufacturer = pd.read_csv(trainTXT, header=None, names =["plane type"])
df_train_manufacturer = df_train_manufacturer.loc[:,"plane type"].str.split(pat = ' ', n=1) #splits image number and manufacture and puts them into list
dftrain = dftrain.join(df_train_manufacturer) #joins boht df into one
#cleaning 
#removing the image/plane id from plane type colum
dftrain['plane type'] = dftrain["plane type"].apply(lambda x:x[1:])
dftrain['plane type'] = dftrain["plane type"].apply(lambda x:"".join(x))
dftrain = dftrain.drop(columns=["Classes", "Labels"]) #remove unnecessary coloums 

print("test data set: ", df.head())
print("")
print("train data set: ", dftrain.head())

#########DISPLAY IMAGES############
# for i in range(1, 26):
#     plt.subplot(5, 5, i)
#     plt.imshow(Image.open(f"/Users/gavinwang/Desktop/CMPM17_ML/CMPM-17-Final-Airplane-Classification/Final Project Data/archive/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images/{df.loc[i, "filename"]}"))
    
#     plt.axis("off")
#     plt.title(df.loc[i, "plane type"])
# plt.tight_layout()
# plt.show()



