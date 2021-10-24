import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms


from models.swin_transformer import SwinTransformer


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


def prepare_data(dataset, dataset_path):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize([224, 224])])

    if dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)
    elif dataset == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader

def load_model(model_path, num_classes):
    
    model = SwinTransformer(num_classes=num_classes) # 修改最后一个线性层的输出单元与类别数一致
    model.cuda()

    if model_path == '':
        print("Training from scratch.")
        return model

    model_dict =  model.state_dict()    
    save_model = torch.load(model_path)['model']

    new_state_dict = {}
    for k, v in save_model.items(): # 从预训练模型读取除最后一个Linear层外的其他部分参数
        if 'head' not in k:
            new_state_dict[k] = v

    model_dict.update(new_state_dict)
    # for k in model_dict:
    #     print("key: ", k, model_dict[k])
    model.load_state_dict(model_dict)

    return model

def warm_up(model, cifar_trainloader, cifar_testloader, warm_up_epochs):
    for k, p in model.named_parameters():
        if "head" not in k:         # 固定除最后一个Linear外的其他参数梯度
            p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0.05) # 过滤，只训练最后一个Linear
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, warm_up_epochs)

    print(len(cifar_trainloader))
    for epoch in range(warm_up_epochs):
        model.train()
        for i, data in enumerate(cifar_trainloader):
            inputs, labels = data
            # print("inputs: ", inputs.size())
    
            optimizer.zero_grad()
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print(i, "  loss: ", loss)
            loss.backward()

            optimizer.step()
            scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in cifar_testloader:
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted==labels).sum().item()

        print("Warm-up epoch {} correct rate: {}".format(epoch, correct / total))

    for p in model.parameters():   # warm-up 结束后，恢复参数梯度
        p.requires_grad = True

if __name__ == '__main__':

    cifar_trainloader, cifar_testloader = prepare_data('CIFAR10', './dataset')
    # cifar_trainloader, cifar_testloader = prepare_data('CIFAR100', './dataset')

    # model = load_model("", num_classes=10)                                              # training from scratch
    model = load_model("save_model/swin_tiny_patch4_window7_224.pth", num_classes=10)   # for finetuning
    
    warm_up(model, cifar_trainloader, cifar_testloader, warm_up_epochs=10)


    epochs = 100
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, epochs)

    file = open("record.txt", "w")
    file.close()
    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(cifar_trainloader):
            inputs, labels = data
            # print("inputs: ", inputs.size())
    
            optimizer.zero_grad()
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print(i, "  loss: ", loss)
            loss.backward()

            optimizer.step()
            scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in cifar_testloader:
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted==labels).sum().item()

        print("Epoch {} correct rate: {}".format(epoch, correct / total))
        with open('record.txt', 'a') as f:
            f.write("Epoch {} correct rate: {}\n".format(epoch, correct / total))