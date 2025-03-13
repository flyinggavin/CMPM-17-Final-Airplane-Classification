import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
from PIL import Image
from hidden import dataset_path, testTXT, dataset_path_train, trainTXT, dataset_path_val, valTXT
import wandb

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
dfall = pd.concat([dftrain, dftest], ignore_index=True)


#####REMOVE MANUFACTURES WE DONT WANT######
#for entire data set
select = []
row = 0
for i in dfall.iterrows():
    if (dfall.loc[row,"plane type"] == "Airbus" or dfall.loc[row,"plane type"] == "Boeing" or dfall.loc[row,"plane type"] == "Bombardier Aerospace" or dfall.loc[row,"plane type"] == "Cessna" or dfall.loc[row,"plane type"] == "Embraer" or dfall.loc[row,"plane type"] == "McDonnell Douglas"):
        select.append(True)
        row += 1
    else:
        select.append(False)
        row += 1
dfall = dfall.loc[select]
print("unique values for dfall:", (dfall["plane type"].unique()))


###FACOTIRIZE PLANE TYPES #### assigning a number to manufactuere 
# dftrain["plane type"] = dftrain["plane type"].factorize()[0]
# dftrain["filename"] = dftrain["filename"].astype(str)
# print("FULL train data set: ")
# print(dftrain.head())
# print(dftrain.info())

# dftest["plane type"] = dftest["plane type"].factorize()[0]
# dftest["filename"] = dftest["filename"].astype(str)

dfall = pd.get_dummies(dfall, columns=["plane type"]) #ONE HOT ENCODED PLANE TYPE INSTEAD OF FACTORIZE

dfall["filename"] = dfall["filename"].astype(str)
print(dfall.shape[0]) # num of rows
print(dfall.size) # tot num of elemnts 


# #####PLOTTING DATA##############
# # for i in range(1, 51):
# #     plt.subplot(5, 10, i)
# #     plt.imshow(Image.open(f"CMPM-17-Final-Airplane-Classification/Final Project Data/fgvc-aircraft-2013b/data/images/{df.loc[i, "filename"]}.jpg"))
# #     plt.axis("off")
# #     plt.title(dftest.loc[i, "Classes"])
# # plt.tight_layout()
# # plt.show()

################IMAGE PROCESSING/ DATA LOADER###############
class MyDataset(Dataset):
    def __init__(self, data_frame):
        self.length = len(data_frame) #return number of rows 
        self.data = data_frame

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        img = Image.open(f"CMPM-17-Final-Airplane-Classification/Final Project Data/fgvc-aircraft-2013b/data/images/{self.data.iloc[idx, 0]}")
        labels = self.data.iloc[idx, 1:7]

        transforms = v2.Compose([
        v2.ToTensor(),
        v2.RandomRotation([-45, 45]),
        v2.RandomGrayscale(),
        v2.GaussianBlur(1),
        v2.Resize([750, 750])
        ])

        img = transforms(img)
        labels = labels.to_numpy(dtype="float64")

        return img, labels

dftrain = dfall.iloc[:3780,:] #70% of 5400 = 3780 
dftest = dfall.iloc[3780:4590, :] #3780+810 = 4590
dfval = dfall.iloc[4590:5401, :] #excludes 5401

#DATA LOADER FOR TRAIN
my_dataset = MyDataset(dftrain)
dataloader_train = DataLoader(my_dataset, batch_size=64, shuffle=True)

#DATA LOADER FOR TEST
my_dataset_test = MyDataset(dftest)
dataloader_test = DataLoader(my_dataset_test, batch_size=32, shuffle=True)

#SPLIT VAL INTO INPUTS(IMAGE) AND OUTPUTS(MANUFACTUER) FOR COMPUTING VAL LOSS
my_dataset_val = MyDataset(dfval)
dataloader_val = DataLoader(my_dataset_val, batch_size = dfval.size, shuffle=True) #dfval.size= total entries (810 x7)

# val_input = dfval.iloc[:,0].to_numpy()
# val_output = dfval.iloc[:,1:].to_numpy()
# val_input = torch.tensor(val_input)
# val_output = torch.tensor(val_output)

#################CNN MODEL###############
class airplaneCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)


        #activiations 
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)

        #Dropout 
        self.dropout = nn.Dropout(p=0.5)

        #Batch Norm
        self.bn1 = nn.BatchNorm1d(750) #750 = number of features 
        

        #linear layers 
        self.linear1 = nn.Linear(65536, 1000)
        self.linear2 = nn.Linear(1000, 2050)
        self.linear3 = nn.Linear(2050, 6)


    def forward(self, input):
        input = self.conv1(input)
        input = self.relu(input)
        input = self.conv2(input)
        input = self.dropout(input) #dropout 
        input = self.relu(input)
        input = self.pool(input)

        input = self.conv3(input)
        input = self.relu(input)
        input = self.pool(input)

        input = input.flatten(start_dim=1)
        input = self.linear1(input)
        input = self.dropout(input) #dropout 
        input = self.relu(input)
        input = self.linear2(input)
        input = self.dropout(input) #dropout 
        input = self.relu(input)
        input = self.linear3(input)
        input = self.relu(input)
        input = self.softmax(input) #softmax

        return input

EPOCHS = 10

model = airplaneCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = ExponentialLR(optimizer, gamma=0.8) #multiplication factor 

run = wandb.init(project="airplane classification", name="run-1") #for wandb

if torch.cuda.is_available():
    device = "cuda"
    print("CUDA Available, using GPU")
else:
    device = "cpu"

model.to(device)

for i in range(EPOCHS):
    print("Epoch", i,)
    loss_sum = 0
    for x, y in dataloader_train:
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        loss_sum += loss
    #VALIDATION  LOSS: #DO we pass val data set into data loader?
    with torch.no_grad():
        for img_val, label_val in dataloader_val:
            val_pred = model.forward(img_val)
            val_loss = loss_fn(val_pred, label_val)
            break
    run.log({"avg train loss":loss_sum/3780, "validation loss":val_loss})

        
   
       
#print("Pred:", pred)
        
        



        
    
