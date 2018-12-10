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

# Plot accuracy
plt.figure(2)
plt.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
plt.xlabel('Number of Iteration')
plt.ylabel('Test Accuracy Values')
plt.title('Test Accuracy')
plt.show()

# plot 20 Kernels for first convoltion layer
nrows = 5
ncols = 4
ker_size = 5
Zero_c= np.zeros((ker_size,1))
Zero_r = np.zeros((1,ker_size+1))
M= np.array([]).reshape(0,ncols*(ker_size+1))

for i in range(nrows):
    N = np.array([]).reshape((ker_size+1),0)

    for j in range(ncols):
        All_kernel = net.params['conv1'][0].data[j + i * ncols][0]

        All_kernel = numpy.matrix(All_kernel)
        All_kernel = np.concatenate((All_kernel,Zero_c),axis=1)
        All_kernel = np.concatenate((All_kernel, Zero_r), axis=0)
        N = np.concatenate((N,All_kernel),axis=1)
    M = np.concatenate((M,N),axis=0)

plt.figure(4)
plt.imshow(M, cmap='Greys',  interpolation='nearest')
plt.title('All Kernels for Conv1')

# Show Caffe architecture
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)






