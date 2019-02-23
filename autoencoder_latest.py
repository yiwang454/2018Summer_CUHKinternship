# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 12:22:03 2018

@author: wangyi66
"""

import os, sys, random, time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# input parameters
data_folder = sys.argv[1]  # the processed folder containing split training and validation sets, and test set
model_folder = sys.argv[2] # folder for saving models and train.log

# hyper paramters
RANDOM_SEED = 0
VALID_PERCENT = 0.1
EPOCH = 100
BATCH_SIZE = 64            
LR = 0.005                 #set learning rate

# other necessary settings
GPU_ID = 0
SAVED = True       
TRAIN_FLAG = True
data_folder_tmp = data_folder.rstrip('/') + '/'
TRAIN_TXT_DATA = data_folder_tmp + 'train_path.txt' # specify a txt file with the path of all training data
TEST_TXT_DATA = data_folder_tmp + 'test_path.txt'

# set random seed
random.seed(RANDOM_SEED)

# create a folder for storing the split training and validation sets
def create_cv_dataset_folder(data_folder, data_folder_tmp):
    try:
        os.mkdir(data_folder)
        os.mkdir(data_folder_tmp + 'train')
        os.mkdir(data_folder_tmp + 'valid')
        os.mkdir(data_folder_tmp + 'test')
    except:
        pass

# get data file paths and write to a txt file
def get_data_paths(folder_path, txt, SAVED):
    if SAVED == False:
        with open(txt, "w") as f_path:
            for root, dirs, files in os.walk(folder_path):
                for name in files:
                    f_path.write(os.path.join(root, name) + '\n')

# link original training data path
def link_original_data_cv(data_folder_tmp, valid_percent = VALID_PERCENT):
    train_valid = []
    train = []
    valid = []
    with open(TRAIN_TXT_DATA) as f_path:
        for line in f_path:
            line = line.strip()
            train_valid.append(line)
    random.shuffle(train_valid)    
    all_num = len(train_valid)
    train_num = int(all_num * (1 - valid_percent))
    train = train_valid[:train_num]
    valid = train_valid[train_num:]
    train_link_folder = data_folder_tmp + 'train/'
    valid_link_folder = data_folder_tmp + 'valid/'
    for item in train:
        train_link = train_link_folder + item.split('/')[-1]
        os.system("ln -s %s %s" % (item, train_link))
    for item in valid:
        valid_link = valid_link_folder + item.split('/')[-1]
        os.system("ln -s %s %s" % (item, valid_link))

# link original test data path
def link_original_data_test(data_folder_tmp):
    test = []
    test_link_folder = data_folder_tmp + 'test/'
    with open(TEST_TXT_DATA) as f_path:
        for line in f_path:
            line = line.strip()
            test_link = test_link_folder + line.split('/')[-1]
            os.system("ln -s %s %s" % (line, test_link))

# get data path after train/validation split
def get_split_train_valid_paths(folder):
    img_paths = []
    files = os.listdir(folder)
    for img in files:
        img_path = folder + img
        img_paths.append(img_path)
    return img_paths

# prepare dataset
def default_loader(path):
    return Image.open(path).convert('L')

class MyDataset(Dataset):
    def __init__(self, paths, transform = None, loader = default_loader):
        imgs = []
        for path in paths:
            imgs.append(path)
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)
    
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__() 
        # encode
        self.encoder = nn.Sequential(
            nn.Linear(128 * 128, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 40),
        )
        # decode
        self.decoder = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 128 * 128),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# calulate the total loss after each epoch
def evaluate(data_loader, device, autoencoder, loss_func):
    total_loss = 0.0
    with torch.no_grad():
        for x in data_loader:
            b_x = x.view(-1, 128 * 128).to(device)
            b_y = x.view(-1, 128 * 128).to(device)
            _, decoded = autoencoder(b_x)
            loss = loss_func(decoded, b_y)
            total_loss += loss
    return total_loss

# training process
def train(train_data_loader, valid_data_loader, device):
    train_log = model_folder + 'train.log'
    save_model_end = model_folder + 'autoencoder.train90.epoch%d.pkl' % EPOCH
    total_time_start = time.time()
    if TRAIN_FLAG:
        # set network and optimization criterion
        autoencoder = AutoEncoder().to(device)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr = LR)
        loss_func = nn.MSELoss().to(device)

        # training process
        for epoch in range(EPOCH):
            epoch_time_start = time.time()
            save_model = model_folder + 'autoencoder.train90.epoch%d.pkl' % (epoch + 1)
            for step, x in enumerate(train_data_loader):
                b_x = x.view(-1, 128 * 128).to(device) # batch to be run into autoencoder
                b_y = x.view(-1, 128 * 128).to(device) # batch for autoencoder output comparison
                
                encoded, decoded = autoencoder(b_x)
                loss = loss_func(decoded, b_y)         # mean square error
                optimizer.zero_grad()                  # clear gradients for this training step
                loss.backward()
                optimizer.step()                       # apply gradients

            total_loss = evaluate(train_data_loader, device, autoencoder, loss_func)
            valid_loss = evaluate(valid_data_loader, device, autoencoder, loss_func)
            epoch_time_end = time.time()
            epoch_time = epoch_time_end - epoch_time_start
            print('Epoch: ', epoch, '| train loss: %.4f' % total_loss.data.cpu().numpy()) 
            print('Epoch: ', epoch, '| valid loss: %.4f' % valid_loss.data.cpu().numpy()) 
            with open(train_log, "a") as record:
                record.write('Epoch: ' + str(epoch) + '| train loss: %.4f' % total_loss.data.cpu().numpy() + '\n')
                record.write('Epoch: ' + str(epoch) + '| valid loss: %.4f' % valid_loss.data.cpu().numpy() + '\n')
                record.write('Epoch: ' + str(epoch) + '| elapsed time: %.4fs' % epoch_time + '\n')
                record.write('\n')
            if (epoch + 1) % 10 == 0:
                torch.save(autoencoder, save_model)
        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        with open(train_log, "a") as record:
            record.write('Total elapsed time: %.4fs' % total_time + '\n') 
        torch.save(autoencoder, save_model_end)

def main():
#    # prepare the training data: split training and validation data, and create soft link
#    create_cv_dataset_folder(data_folder, data_folder_tmp) 
#    get_data_paths(train_folder, TRAIN_TXT_DATA, SAVED = False)
#    link_original_data_cv(data_folder_tmp)
#
#    # prepare the test data: create soft link
#    get_data_paths(test_folder, TEST_TXT_DATA, SAVED = False)
#    link_original_data_test(data_folder_tmp)

    # check gpu availability
    if torch.cuda.is_available():
        torch.cuda.set_device(GPU_ID)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    global model_folder
    try:
        os.mkdir(model_folder)
    except:
        pass 
    model_folder = model_folder.rstrip('/') + '/'

    # get train/validation spli set paths
    train_paths = get_split_train_valid_paths(data_folder_tmp + 'train/')
    valid_paths = get_split_train_valid_paths(data_folder_tmp + 'valid/')  

    # load training and validation data
    train_data = MyDataset(paths = train_paths,  transform = transforms.ToTensor())
    valid_data = MyDataset(paths = valid_paths, transform = transforms.ToTensor())
    train_data_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
    valid_data_loader = DataLoader(valid_data, batch_size = BATCH_SIZE, shuffle = True)

    # training without validation set
    train(train_data_loader, valid_data_loader, device)

if __name__ == '__main__':
    main()
