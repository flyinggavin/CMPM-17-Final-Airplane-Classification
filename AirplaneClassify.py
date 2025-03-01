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

# for i in range(1, 51):
#     plt.subplot(5, 10, i)
#     plt.imshow(Image.open(f"CMPM-17-Final-Airplane-Classification/Final Project Data/fgvc-aircraft-2013b/data/images/{df.loc[i, "filename"]}.jpg"))
#     plt.axis("off")
#     plt.title(df.loc[i, "Classes"])
# plt.tight_layout()
# plt.show()

# class MyDataset(Dataset):
#     def __init__(self):
#         self.length = len(df)
#         self.data=df
#     def __len__(self):
#         return self.length``
#     def __getitem__(self, idx):
#         inputCols = [0]
#         outputCols = [1]
        
#         input = self.data.iloc[idx, inputCols]
#         output = self.data.iloc[idx, outputCols]

#         input = input.to_numpy(dtype="float64")
#         output = output.to_numpy(dtype="float64")

#         return input, output
    
# my_dataset = MyDataset()

# dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)
# for train_x, train_y in dataloader:
#      print(train_x.shape, train_y.shape)



