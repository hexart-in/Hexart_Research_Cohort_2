from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset
from PIL import Image
from sklearn.metrics import cohen_kappa_score


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


class dr_dataset(Dataset):

    def __init__(self, numpy_array, data_dir, transform=None, target_transform=None, ext=".png"):
        self.img_labels = pd.DataFrame(numpy_array)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.ext = ext



    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, item):
        img_path = os.path.join(self.data_dir,self.img_labels.iloc[item,0]+self.ext)
        img = Image.open(img_path)
        label = self.img_labels.iloc[item,1]
        if self.transform:
            img = self.transform(img)
        else:
            self.transform = transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return (img, label,)

def get_sample(dataset:str) -> np.ndarray:
    """
    This function will sample unbalanced dr_dataset into a balanced dr_dataset
    ***FOR DIABETIC RETINPATHY ONLY***
    :param dataset: csv file of the dr_dataset
    :return: numpy array containing shuffled balanced data samples
    """

    df_data = pd.read_csv(dataset)
    df_data_classes = dict.fromkeys(pd.unique(df_data.diagnosis))
    for i in pd.unique(df_data.diagnosis):
        df_data_classes[i] = df_data[df_data.diagnosis == i]


    n = np.min((df_data_classes[0].shape[0],
                df_data_classes[1].shape[0],
                df_data_classes[2].shape[0],
                df_data_classes[3].shape[0],
                df_data_classes[4].shape[0],))

    n = int(n - (n//4))

    rng = np.random.default_rng()
    train_samples = np.concatenate((rng.permutation(df_data_classes[0])[:n],
                                   rng.permutation(df_data_classes[1])[:n],
                                   rng.permutation(df_data_classes[2])[:n],
                                   rng.permutation(df_data_classes[3])[:n],
                                   rng.permutation(df_data_classes[4])[:n],))

    random_training_samples = rng.permutation(train_samples)

    return random_training_samples

path = "/home/hexart/dataset/aptos2019-blindness-detection"
random_training_samples = get_sample(os.path.join(path, 'train.csv'))
training_set = random_training_samples[:int(len(random_training_samples)*0.8)]
testing_set = random_training_samples[int(len(random_training_samples) * 0.8):]

print("GPU AVAILABLE: {}".format(torch.cuda.is_available()))

image_dataset = dict.fromkeys(['train','val'])
image_dataset['train'] = dr_dataset(numpy_array=training_set,
                                     data_dir=os.path.join(path,'train_images'),
                                     transform = data_transforms['train'])
image_dataset['val'] = dr_dataset(numpy_array=testing_set,
                                     data_dir=os.path.join(path,'train_images'),
                                     transform = data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_dataset[x],
                                              batch_size=12,
                                              shuffle=True,
                                              num_workers=2)
               for x in ['train','val']}
dataset_sizes = {x: len(image_dataset[x]) for x in ['train', 'val']}
class_names = {
    0 : "No DR",
    1 : "NPDR Mild",
    2 : "NPDR Moderate",
    3 : "NPDR Severe",
    4 : "PDR"
}


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')




def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    """
    print("TRAINING STARTED WITH: \nModel: {}\nCRITERION: {}\nOPTIMIZER: {}\nSCHEDULER: {}\nEPOCS: {}\n".format(
        model, criterion, optimizer, scheduler, num_epochs,
    ))
    """

    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_kappa = 0.0
    """
    train_metrics = []
    val_metrics = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            predictions = []
            ground_truth = []
            #print("PHASE: {}".format(phase))

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0
            #count = 0
            #print("{}".format(count))

            for inputs, labels in dataloaders[phase]:
                #count = count + 1
                #print("{}.".format(count), end="")
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # print("Labels:\t {},\t{}".format(labels, type(labels)))
                    # print("Preds:\t {},\t{}".format(preds, type(preds)))
                    # print("Loss:\t {:.4f}".format(loss))
                    # This is where we need to install the quadratic kappa loss

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
                predictions.extend(preds.cpu().detach().numpy())
                ground_truth.extend(labels.cpu().detach().numpy())

            if phase == 'train':
                scheduler.step()

            """
            print("\nPreds:\t{}\t".format(len(predictions)))
            print("Labels:\t{}\t".format(len(ground_truth)))
            """
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_kappa = cohen_kappa_score(predictions, ground_truth)
            print('\n{} Loss: {:.4f} Acc: {:.4f} Kappa: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_kappa))
            if phase == 'train':
                train_metrics.append([epoch, epoch_loss, epoch_acc, epoch_kappa])
            else:
                val_metrics.append([epoch, epoch_loss, epoch_acc, epoch_kappa])

    time_elapsed = time.time() - since
    print("Training Completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    return (model, train_metrics, val_metrics)




device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 5)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    #optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model, train_metrics, val_metrics = train_model(model_ft,
                                                    criterion,
                                                    optimizer_ft,
                                                    exp_lr_scheduler,num_epochs=10)
    print(train_metrics)
    print(val_metrics)




