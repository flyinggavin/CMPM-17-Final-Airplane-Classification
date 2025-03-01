import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
from PIL import Image
from hidden import dataset_path, testTXT, dataset_path_train, trainTXT, dataset_path_val, valTXT


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


####VAL DATA SET########
dfval = pd.read_csv(dataset_path_val)
df_val_manufacturer = pd.read_csv(valTXT, header=None, names =["plane type"])
df_val_manufacturer = df_val_manufacturer.loc[:,"plane type"].str.split(pat = ' ', n=1)
dfval = dfval.join(df_val_manufacturer)

dfval['plane type'] = dfval["plane type"].apply(lambda x:x[1:])
dfval['plane type'] = dfval["plane type"].apply(lambda x:"".join(x))
dfval = dfval.drop(columns=["Classes", "Labels"]) #remove unnecessary coloums 

print("test data set: ", df.head())
print("")
print("val data set: ", dfval.head())
print("")
print("train data set: ", dftrain.head())
print(dftrain.info())
print("")


#######COMBINED VAL WITH TRAIN TO MAKE FULL TRIAN DATA SET##############
dftrain = pd.concat([dftrain, dfval], ignore_index=True)

###FACOTIRIZE PLANE TYPES #### for training
dftrain["plane type"] = dftrain["plane type"].factorize()[0]
dftrain["filename"] = dftrain["filename"].astype("string")
print("FULL train data set: ")
print(dftrain.head())
print(dftrain.info())



# print(df.shape[0]) # num of rows
# print(df.size) # tot num of elemnts 

#####PLOTTING DATA##############
# for i in range(1, 51):
#     plt.subplot(5, 10, i)
#     plt.imshow(Image.open(f"CMPM-17-Final-Airplane-Classification/Final Project Data/fgvc-aircraft-2013b/data/images/{df.loc[i, "filename"]}.jpg"))
#     plt.axis("off")
#     plt.title(df.loc[i, "Classes"])
# plt.tight_layout()
# plt.show()

#################IMAGE PROCESSING/ DATA LOADER###############
class MyDataset(Dataset):
    def __init__(self):
        self.length = df.shape[0] #return numbe rof rows 
        self.data = df

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        output = self.data.loc[idx, "plane type"]
        input = self.data.loc[idx, "filename"]
        return input, output 
    
        # inputCols = [0]
        # outputCols = [1]
        
        # input = self.data.iloc[idx, inputCols]
        # output = self.data.iloc[idx, outputCols]

        # input = input.to_numpy(dtype="float64")
        # output = output.to_numpy(dtype="float64")

        # return input, output
    
my_dataset = MyDataset()
dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)

for x, y in dataloader:
     print(x.shape())
     print(y.shape())



