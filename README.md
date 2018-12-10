# Airbus Ship Detection
* 1) Download the repository
* 2) Unzip the csv file
* 3) Run the get_data.py file to get the images
* 4) In you current working directory, create a train and validation folder. Within each folder, create subdirectories 00000 and 00001. This is for creating the lmdb for Caffe
* 5) Run the preprocess.py file. Now you should see 50,000 images in the train folder and 10,000 in the validation folder
* 6) Run the create_lmdb. py file--you should see train_lmdb and validation lmdb show up
* 7) Run the airbus_train.py file
* NOTE: Keep in mind that all of these files, including the data files, should be in the same directory
