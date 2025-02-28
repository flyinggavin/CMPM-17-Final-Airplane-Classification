import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
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
df_test_manufacturer = pd.read_csv(testTXT)
print(df_test_manufacturer.info())

#remove the .jpg from filename of the df so ID can match with manufacture df 
df = df.replace(".jpg","",regex=True)
print(df.head()) 

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





