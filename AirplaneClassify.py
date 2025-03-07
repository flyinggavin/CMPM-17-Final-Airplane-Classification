import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
from PIL import Image
from hidden import dataset_path, testTXT, dataset_path_train, trainTXT, dataset_path_val, valTXT


##########TEST DATA##################
#ClEANING DATA SET
dftest = pd.read_csv(dataset_path) #test data 

#read the test manufacture text as a dataframe 
df_test_manufacturer = pd.read_csv(testTXT, header=None, names =["plane type"])

#remove the .jpg from filename of the df so ID can match with manufacture df 
#df = df.replace(".jpg","",regex=True)
df_test_manufacturer = df_test_manufacturer.loc[:,"plane type"].str.split(pat = ' ', n=1) #split space since txt uses sapce

#combines manufactuer with df 
dftest = dftest.join(df_test_manufacturer)


#removing the image/plane id from plane type colum
dftest['plane type'] = dftest["plane type"].apply(lambda x:x[1:])
dftest['plane type'] = dftest["plane type"].apply(lambda x:"".join(x))

dftest = dftest.drop(columns=["Classes", "Labels"]) #remove unnecessary coloums 


print(dftest.head())

##########TRAIN DATA##################
dftrain = pd.read_csv(dataset_path_train) #trial dataset
df_train_manufacturer = pd.read_csv(trainTXT, header=None, names =["plane type"])
df_train_manufacturer = df_train_manufacturer.loc[:,"plane type"].str.split(pat = ' ', n=1) #splits image number and manufacture and puts them into list
dftrain = dftrain.join(df_train_manufacturer) #joins both df into one
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

print("test data set: ", dftest.head())
print("")
print("val data set: ", dfval.head())
print("")
print("train data set: ", dftrain.head())
print(dftrain.info())
print("")


#######COMBINED VAL WITH TRAIN TO MAKE FULL TRIAN DATA SET##############
dftrain = pd.concat([dftrain, dfval], ignore_index=True)


#####REMOVE MANUFACTURES WE DONT WANT######

#for test data set
select = []
row = 0
for i in dftest.iterrows():
    if (dftest.loc[row,"plane type"] == "Airbus" or dftest.loc[row,"plane type"] == "Boeing" or dftest.loc[row,"plane type"] == "Bombardier Aerospace" or dftest.loc[row,"plane type"] == "Cessna" or dftest.loc[row,"plane type"] == "Embraer" or dftest.loc[row,"plane type"] == "McDonnell Douglas"):
        select.append(True)
        row += 1
    else:
        select.append(False)
        row += 1
dftest = dftest.loc[select]

#for train data set
select = []
row = 0
for i in dftrain.iterrows():
    if (dftrain.loc[row,"plane type"] == "Airbus" or dftrain.loc[row,"plane type"] == "Boeing" or dftrain.loc[row,"plane type"] == "Bombardier Aerospace" or dftrain.loc[row,"plane type"] == "Cessna" or dftrain.loc[row,"plane type"] == "Embraer" or dftrain.loc[row,"plane type"] == "McDonnell Douglas"):
        select.append(True)
        row += 1
    else:
        select.append(False)
        row += 1
dftrain = dftrain.loc[select]

print("unique train vals:", (dftest["plane type"].unique()))
print("unique tests vals:", (dftest["plane type"].unique()))


###FACOTIRIZE PLANE TYPES #### for training
dftrain["plane type"] = dftrain["plane type"].factorize()[0]
dftrain["filename"] = dftrain["filename"].astype(str)
print("FULL train data set: ")
print(dftrain.head())
print(dftrain.info())

dftest["plane type"] = dftest["plane type"].factorize()[0]
dftest["filename"] = dftest["filename"].astype(str)


print(dftest.shape[0]) # num of rows
print(dftest.size) # tot num of elemnts 

#####PLOTTING DATA##############
# for i in range(1, 51):
#     plt.subplot(5, 10, i)
#     plt.imshow(Image.open(f"CMPM-17-Final-Airplane-Classification/Final Project Data/fgvc-aircraft-2013b/data/images/{df.loc[i, "filename"]}.jpg"))
#     plt.axis("off")
#     plt.title(dftest.loc[i, "Classes"])
# plt.tight_layout()
# plt.show()

################IMAGE PROCESSING/ DATA LOADER###############
class MyDataset(Dataset):
    def __init__(self, data_frame):
        self.length = len(data_frame) #return number of rows 
        self.data = data_frame

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        img = Image.open(f"CMPM-17-Final-Airplane-Classification/Final Project Data/archive/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images/{self.data.iloc[idx, 0]}")
        label = self.data.iloc[idx, [1]]

        transforms = v2.Compose([
        v2.ToTensor(),
        v2.RandomRotation([-45, 45]),
        v2.RandomGrayscale(),
        v2.GaussianBlur(1),
        v2.Resize([1000, 750])
        ])

        img = transforms(img)
        label = label.to_numpy(dtype="float64")

        return img, label
    
my_dataset = MyDataset(dftrain)
dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)

# for x, y in dataloader:
#      print(x.shape)
#      print(y.shape)

my_dataset_test = MyDataset(dftest)
dataloader_test = DataLoader(my_dataset_test, batch_size=32, shuffle=True)

# for x, y in dataloader_test:
#      print(x.shape)
#      print(y.shape)

#################CNN MODEL###############
class airplaneCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.linear1 = nn.Linear(32 * 1000 * 750, 1000)
        self.linear2 = nn.Linear(1000, 2050)
        self.linear3 = nn.Linear(2050, 10)

    def forward(self, input):
        input = self.conv1(input)
        input = self.relu(input)
        input = self.conv2(input)
        input = self.relu(input)
        input = self.pool(input)

        input = self.conv3(input)
        input = self.relu(input)
        input = self.pool(input)

        input = input.flatten(start_dim=1)
        input = self.linear1(input)
        input = self.relu(input)
        input = self.linear2(input)
        input = self.relu(input)
        input = self.linear3(input)
        input = self.relu(input)

        return input
