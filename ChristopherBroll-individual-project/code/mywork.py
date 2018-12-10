
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
# -----------------------------------------------------------------------------------------
# NOTE: this is based on the train_mnist.py file from amir-jafari on Github
# Source: https://github.com/amir-jafari/Deep-Learning/blob/master/Caffe_/3-Create_LMDB/create_lmdb_tutorial.py

import os
import matplotlib
matplotlib.use('Qt4Agg') # Used this to export displays
import caffe
import matplotlib.pyplot as plt
import numpy as np

# Use GPU
caffe.set_mode_gpu()

# Use SGD Solver for momentum
solver = caffe.SGDSolver('airbus_solver.prototxt')

# Use 5000 to stop overfitting
niter = 5000
test_interval = 100

# Initialize train_loss, val_loss and test_acc to plot each later
train_loss = np.zeros(niter)
val_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data

    # store the validation loss
    val_loss[it] = solver.test_nets[0].blobs['loss_val'].data

    solver.test_nets[0].forward(start='conv1')

    if it % test_interval == 0:
        acc=solver.test_nets[0].blobs['accuracy'].data
        print 'Iteration', it, 'testing...','accuracy:',acc
        test_acc[it // test_interval] = acc


# Plot training loss vs validation to check for overfitting
plt.figure(1)
p1 = plt.plot(np.arange(niter), train_loss)
p2 = plt.plot(np.arange(niter), val_loss)
plt.xlabel('Number of Iteration')
plt.ylabel('Loss Values')
plt.title('Training vs Validation Loss')
plt.legend((p1[0], p2[0]), ('Train', 'Validation'))
# -----------------------------------------------------------------------------
import os

os.system("wget https://storage.googleapis.com/cjbroll_ml2_final_project_data/data_full.zip")
os.system("unzip data_full.zip")
# -----------------------------------------------------------------------------------
name: "LeNet for Airbus Ship Detection Challenge"
layer {
  name: "airbus"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "train_lmdb"
    batch_size: 64
    backend: LMDB
  }
}
layer {
  name: "airbus"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "validation_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 60
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 5
    stride: 5
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 120
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 5
    stride: 5
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "pool2"
  top: "ip1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 1000
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"
  top: "ip1"
}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "ip2"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }

}
layer {
  name: "loss_val"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss_val"
  include {
    phase: TEST
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}

