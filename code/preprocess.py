#NOTE: We borrowed code from Siddartha's kernel "Airbus Ship Detection" (first 18 lines)
# Source: https://www.kaggle.com/meaninglesslives/airbus-ship-detection-data-visualization
# ----------------------------------------------------------------------------------------------------------------------

# import packages
import numpy as np
import pandas as pd
from skimage.data import imread
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# Read data
df = pd.read_csv('../ml2_final/train_ship_segmentations_v2.csv')

# Adding shipcount column
df = df.reset_index()
df['ship_count'] = df.groupby('ImageId')['ImageId'].transform('count')
df.loc[df['EncodedPixels'].isnull().values,'ship_count'] = 0

#----------------------------- Preprocess (End of Siddartha's code) ---------------------------------------------------

masks.loc[masks.ship_count > 1, 'ship_count'] = 1
masks = masks[['ImageId','ship_count']]
masks = masks.drop_duplicates()

# Get Images and labels
X = masks.ImageId.values
y = masks.ship_count.values

# Get subset of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.3, stratify=y_train)

# Split into training and validation image sets (50,000 and 10,000, respectively)
train_df = pd.DataFrame({"ImageID": X_train2[:50000], "label": y_train2[:50000]})
val_df = pd.DataFrame({"ImageID": X_test2[:10000], "label": y_test2[:10000]})

# Seperate labels for subfolders for Caffe CNN
train_df_lab_1 = train_df[train_df.label==1]
train_df_lab_0 = train_df[train_df.label==0]

val_df_lab_1 = val_df[val_df.label==1]
val_df_lab_0 = val_df[val_df.label==0]

#--------------------------- Prepare folders for Caffe -----------------------------------------------------------------

# train folder with label 1 (images with ships)
for file in train_df_lab_1.ImageID.tolist():
    try:
        os.rename("../ml2_final/train_v2/" + file, "../ml2_final/train/00001/" + file)
    except:
        print('No')

# train folder with label 0 (images with no ships)
for file in train_df_lab_0.ImageID.tolist():
    try:
        os.rename("../ml2_final/train_v2/" + file, "../ml2_final/train/00000/" + file)
    except:
        print('No')

# validation folder with label 1 (images with ships)
for file in val_df_lab_1.ImageID.tolist():
    try:
        os.rename("../ml2_final/train_v2/" + file, "../ml2_final/validation/00001/" + file)
    except:
        print('No')
# validation folder with label 0 (images with no ships)
for file in val_df_lab_0.ImageID.tolist():
    try:
        os.rename("../ml2_final/train_v2/" + file, "../ml2_final/validation/00000/" + file)
    except:
        print("No")