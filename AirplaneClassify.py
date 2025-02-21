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
df_test_manufacturer = pd.read_csv(testTXT)
print(df_test_manufacturer.info())

#remove the .jpg from filename of the df so ID can match with manufacture df 
df = df.replace(".jpg","",regex=True)
print(df.head()) 

for i in range(1, 26):
    plt.subplot(5, 5, i)
    plt.imshow(Image.open(f"CMPM-17-Final-Airplane-Classification/Final Project Data/fgvc-aircraft-2013b/data/images/{df.loc[i, "filename"]}.jpg"))
    plt.axis("off")
    plt.title(df.loc[i, "Classes"])
plt.tight_layout()
plt.show()




