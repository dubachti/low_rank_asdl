import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
from utils import progress_bar
import low_rank_asdl as asdl

def main():
    parser = argparse.ArgumentParser(description='ResNet-18 CIFAR-10 Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=5, type=int, help='num epochs')
    parser.add_argument('--ft', default='fisher_mc', type=str, help='fisher type')
    parser.add_argument('--fs', default='kron_lr', type=str, help='fisher shape')
    parser.add_argument('--damp', default=1e-2, type=float, help='damping value')
    parser.add_argument('--rank', default=1, type=int, help='rank size')
    parser.add_argument('--itr', default=1, type=int, help='max power iterations')
    parser.add_argument('--interval', default=50, type=int, help='inverse refresh rate')
    parser.add_argument('--b_size', default=128, type=int, help='batch size')
    parser.add_argument('--ignore', default=True, type=bool, help='ignore first and last module')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.b_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    net = models.resnet18() # in-place operations for torch.models have to be disabled
    net.fc = torch.nn.Linear(512, 10) # adjust last layer to size 10
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3)
    ignore_modules = [nn.BatchNorm2d] # batch-norm not supported

    # ignore first and last layer
    if args.ignore:
        fisher_shapes = [
            ('conv1',               'kron'),
            ('fc',                  'kron'),
            f'{args.fs}',
        ]
    else: fisher_shapes = args.fs

    ngd = asdl.NaturalGradient(net, fisher_type=args.ft, fisher_shape=fisher_shapes, 
                               ignore_modules=ignore_modules, rank=args.rank, max_itr=args.itr)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            if not batch_idx%args.interval==0:
                loss, outputs = ngd.update_curvature(inputs=inputs, targets=targets,
                                                     calc_emp_loss_grad=True, new_curvature=False, 
                                                     accumulate=True)
            else:
                loss, outputs = ngd.update_curvature(inputs=inputs, targets=targets,
                                                     calc_emp_loss_grad=True, new_curvature=True, 
                                                     accumulate=False)
                ngd.update_inv(damping=args.damp)
            ngd.precondition()
            optimizer.step()

            train_loss += loss
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum()
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test(epoch):
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


    test(0)
    for epoch in range(args.epoch):
        train(epoch)
        test(epoch)
        scheduler.step()

if __name__ == '__main__': main()
