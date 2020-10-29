from medmnist.models import ResNet18, ResNet50
from medmnist.dataset import INFO, PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, BreastMNIST, OrganMNIST_Axial, OrganMNIST_Coronal, OrganMNIST_Sagittal
from medmnist.environ import outputroot
from medmnist.evaluator import getAUC, getACC, save

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms


def main(flag):
    ''' main function
    :param flag: name of subset

    '''

    dataclass = {
        "pathmnist": PathMNIST,
        "chestmnist": ChestMNIST,
        "dermamnist": DermaMNIST,
        "octmnist": OCTMNIST,
        "pneumoniamnist": PneumoniaMNIST,
        "retinamnist": RetinaMNIST,
        "breastmnist": BreastMNIST,
        "organmnist_axial": OrganMNIST_Axial,
        "organmnist_coronal": OrganMNIST_Coronal,
        "organmnist_sagittal": OrganMNIST_Sagittal,
    }

    with open(INFO, 'r') as f:
        info = json.load(f)
        task = info[flag]['task']
        n_channels = info[flag]['n_channels']
        n_classes = len(info[flag]['label'])

    start_epoch = 0
    end_epoch = 99
    lr = 0.001
    batch_size = 128
    val_auc_list = []

    print('==> Preparing data..')
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_dataset = dataclass[flag](split='train', transform=train_transform)
    train_loader = data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = dataclass[flag](split='val', transform=val_transform)
    val_loader = data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = dataclass[flag](split='test', transform=test_transform)
    test_loader = data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True)

    print('==> Building model..')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet18(in_channels=n_channels, num_classes=n_classes).to(device)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(start_epoch, end_epoch + 1):
        train(model, optimizer, criterion, train_loader, device, task)
        val(model, val_loader, device, val_auc_list, flag, task, epoch)
    
    auc_list = np.array(val_auc_list)
    index = auc_list.argmax()
    print('epoch %s is the best model' % (index))

    restore_model_path = 'checkpoints_ResNet18/%s_checkpoints/ckpt_%d_auc_%.5f.pth' % (flag, index, auc_list[index])
    model.load_state_dict(torch.load(restore_model_path)['net'])
    test(model, 'train', train_loader, device, flag, task)
    test(model, 'val', val_loader, device, flag, task)
    test(model, 'test', test_loader, device, flag, task)


def train(model, optimizer, criterion, train_loader, device, task):
    ''' training function
    :param model: the model to train
    :param optimizer: optimizer used in training
    :param criterion: loss function
    :param train_loader: DataLoader of training set
    :param device: cpu or cuda
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class

    '''

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long().to(device)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()


def val(model, val_loader, device, val_auc_list, flag, task, epoch):
    ''' validation function
    :param model: the model to validate
    :param val_loader: DataLoader of validation set
    :param device: cpu or cuda
    :param val_auc_list: the list to save AUC score of each epoch
    :param flag: subset name
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class
    :param epoch: current epoch

    '''

    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax()
                outputs = m(outputs).to(device)
                targets = targets.resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        val_auc_list.append(auc)

    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch,
    }
    dir_path = 'checkpoints_ResNet18/%s_checkpoints' % (flag)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
    torch.save(state, path)


def test(model, split, data_loader, device, flag, task):
    ''' testing function
    :param model: the model to test
    :param split: the data to test, 'train/val/test'
    :param data_loader: DataLoader of data
    :param device: cpu or cuda
    :param flag: subset name
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class

    '''

    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax()
                outputs = m(outputs).to(device)
                targets = targets.resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        acc = getACC(y_true, y_score, task)
        print('%s AUC: %.5f ACC: %.5f' % (split, auc, acc))

        outputdir = os.path.join(outputroot, flag)
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        outputpath = os.path.join(outputdir, '%s.csv' % (split))
        save(y_true, y_score, outputpath)


if __name__ == '__main__':
    data_name = sys.argv[1]
    main(data_name)
