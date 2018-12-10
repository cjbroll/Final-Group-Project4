import os
import wget

# If already exsit previous lmdb folders, remove them
os.system('rm -rf  train_lmdb')
os.system('rm -rf  validation_lmdb')

os.mkdir('train_lmdb')
os.mkdir('validation_lmdb')

url1 = 'https://storage.googleapis.com/cjbroll_ml2_final_project_data/data_lmdb/train_lmdb/data.mdb'
url2 = 'https://storage.googleapis.com/cjbroll_ml2_final_project_data/data_lmdb/train_lmdb/lock.mdb'
wget.download(url1, './train_lmdb/')
wget.download(url2, './train_lmdb/')

url3 = 'https://storage.googleapis.com/cjbroll_ml2_final_project_data/data_lmdb/validation_lmdb/data.mdb'
url4 = 'https://storage.googleapis.com/cjbroll_ml2_final_project_data/data_lmdb/validation_lmdb/lock.mdb'
wget.download(url3, './validation_lmdb/')
wget.download(url4, './validation_lmdb/')