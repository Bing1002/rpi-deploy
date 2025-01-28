# Train CIFAR10 with PyTorch.
# torchrun --standalone --nproc_per_node=4 main_ddp.py

# Train tricks:
#   weight init 
#   lr and lr scheduler 
#   batch size
#   batch normalization, layer normalization, group normalization 
#   optimizer: SGD, Adam, RMSprop, AdamW, Muon 
#   data augmentation:  
#   early stop
#   gradient clipping 
#   EMA
#   DDP 
#   Gradient accumulation (microbatch): to mimic large batches 
#   Warmup 
#   gradient, activation check 
#   
import os
import json 
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from resnet import ResNet18
from utils import progress_bar


# ddp 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
else:
    # if not ddp, we are running on a single gpu, and one process
    ddp_rank = 0
    ddp_local_rank = 0
    zero_stage = 0
    ddp_world_size = 1
    master_process = True
    seed_offset = 0
    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda"


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform_train)
if ddp:
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=256, 
        # shuffle=True, 
        shuffle=False,
        num_workers=2, 
        sampler=DistributedSampler(trainset, shuffle=True))  # default is True
else:
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=256, 
        shuffle=True, 
        num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=100, 
    shuffle=False, 
    num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)

if ddp:
    net = DDP(net, device_ids=[ddp_local_rank])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

log_stats = defaultdict(dict)
# Training
def train(epoch):
    if master_process:
        print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if master_process:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    #
    if master_process:
        log_stats['train'][epoch] = [train_loss/(batch_idx+1), 100.*correct/total, correct, total]


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
        log_stats['test'][epoch] = [test_loss/(batch_idx+1), 100.*correct/total, correct, total]

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(0, 150):
    if ddp:
        trainloader.sampler.set_epoch(epoch)
    train(epoch)
    
    # only run test on the rank 0 nodel 
    if ddp_local_rank == 0 and master_process:
        test(epoch)
    scheduler.step()

if master_process:
    print('best acc: ', best_acc)
    with open('log_stats.json', 'w') as f:
        json.dump(log_stats, f)

if ddp:
    destroy_process_group()