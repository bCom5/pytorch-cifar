'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar, profile

model_dict = {
    'squeezenet': lambda:SqueezeNet(num_classes=10),
    'mobilenet': lambda: MyMobileNet(),
    'mobilenetv2': lambda: MobileNetV2(),
    'mobilenet_small': lambda: MyMobileNet(width_mul=.25),
    'fd_mobilenet': lambda: MyMobileNet(is_fd=True),
    'fd_mobilenet_small': lambda: MyMobileNet(width_mul=.25, is_fd=True),
    'mobilenetv3_small_x0.35': lambda: MobileNetV3(n_class=10, width_mult=.35),
    'mobilenetv3_small_x0.75': lambda: MobileNetV3(n_class=10, width_mult=.75),
    'mobilenetv3_impl2_small_x1.00': lambda: MobileNetV3Imp2(classes_num=10, input_size=32, width_multiplier=1.00, mode='small'),
    'mobilenetv3_impl2_small_x0.25': lambda: MobileNetV3Imp2(classes_num=10, input_size=32, width_multiplier=0.25, mode='small'),
    'fd_mobilenet_impl2_small_x0.25': lambda: FdMobileNetV3Imp2(classes_num=10, input_size=32, width_multiplier=0.25, mode='ours1'),
    'fd_mobilenet_impl2_small_x0.32': lambda: FdMobileNetV3Imp2(classes_num=10, input_size=32, width_multiplier=0.32, mode='ours1'),
    'fd_mobilenet_impl2_small_x1.00': lambda: FdMobileNetV3Imp2(classes_num=10, input_size=32, width_multiplier=1.00, mode='ours1'),
    'ours2_x0.25': lambda: FdMobileNetV3Imp2(classes_num=10, input_size=32, width_multiplier=0.25, mode='ours2'),
    'ours2_x1.00': lambda: FdMobileNetV3Imp2(classes_num=10, input_size=32, width_multiplier=1.00, mode='ours2'),
}
'''
TODO
Note this
'vgg': VGG('VGG19')
'resnet': ResNet18()
'preact_resnet': PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()

# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
'''

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--net', default='mobilenet', choices=list(model_dict), help='neural net model to run')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--auto', dest='auto', action='store_true')
parser.add_argument('--rmsauto', dest='rmsauto', action='store_true')
args = parser.parse_args()

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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

get_model = model_dict[args.net]
net = get_model()
print('==> Model:', args.net)
flops, params = profile(net, inputs=(torch.randn(1, 3, 32, 32), ))
print('* MACs: {:,.2f}'.format(flops).replace('.00', ''))
print('* Params: {:,.2f}'.format(params).replace('.00', ''))

if torch.cuda.is_available():
    device = 'cuda'
    print('==> cuda is available (gpu)')
else:
    device = 'cpu'
    print('==> No cuda, running on cpu')
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) # RMSprop gradients explode??
state = {
    'net': net.state_dict()
}
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # print('just check', optimizer)
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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

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

if args.rmsauto:
    lr = .005
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=5e-4)
    for epoch in range(start_epoch, 160):
        if epoch == 40:
            net.load_state_dict(state['net'])
            lr  = .001
            optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=5e-4)
        elif epoch == 80:
            net.load_state_dict(state['net'])
            lr = .0005
            optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=5e-4)
        elif epoch == 120:
            net.load_state_dict(state['net'])
            lr = .0001
            optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=5e-4)
        train(epoch)
        test(epoch)

elif args.auto:
    lr = .1
    for epoch in range(start_epoch, 200):
        if epoch == 50:
            net.load_state_dict(state['net'])
            lr  = .01
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        elif epoch == 100:
            net.load_state_dict(state['net'])
            lr = .001
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        elif epoch == 150:
            net.load_state_dict(state['net'])
            lr = .0001
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        train(epoch)
        test(epoch)

else:
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)

