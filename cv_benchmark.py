'''
Copied and modified from
https://github.com/pytorch/examples/blob/master/mnist/main.py
'''

import argparse
import numpy as np
import os
import statistics
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from datetime import datetime
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR

# from . import cifar_architecture
from cifar_architecture import resnet
from cifar_architecture import vgg
from cifar_architecture import squeezenet
from cifar_architecture import senet
from cifar_architecture import densenet
import resnet_akamaster


try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example.")


def train(args, model, device, train_loader, optimizer, epoch, scaler=None, enable_amp_autocast=False):
    """Trains for a single epoch

    Args:
        args (Namespace): Arguments
        model ([type]): [description]
        device ([type]): [description]
        train_loader (DataLoader): DataLoader
        optimizer ([type]): [description]
        epoch (int): Epoch id
        scaler ([type], optional): Passed in if using torch.cuda.amp for quantization. Defaults to None.
        enable_amp_autocast (bool, optional): True if using torch.cuda.amp for quantization in the 
        current epoch. Defaults to False.
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx == args.breakpoint:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Hitting breakpoint, exiting", flush=True)
            exit(0)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=enable_amp_autocast):
            output = model(data)
            criterion = nn.CrossEntropyLoss().to(device)
            loss = criterion(output, target)
        if args.cuda_amp:  # use torch.cuda.amp for quantization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        elif args.apex_amp is not None or args.apex_amp_dynamic is not None:  # use apex.amp with specified level
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        else:  # no quantization
            loss.backward()
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            # test_loss += F.nll_loss(output, target, reduction='sum').item()
            criterion = nn.CrossEntropyLoss().to(device)
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print('\n[{}] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        current_time,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def get_model(args):
    # choosing the right model is important
    # imagenet: 224*224 input, use default model from torchvision
    # SVHN/CIFAR10/CIFAR100: 32*32 input, use model from cifar_architecture
    if args.model in ["resnet20", "resnet32", "resnet44", "resnet56", "resnet110"]:
        model = getattr(resnet_akamaster, args.model)()
    elif args.dataset == "imagenet":
        model = getattr(models, args.model)(num_classes=args.num_classes)
    elif "resnet" in args.model:
        model = getattr(resnet, args.model)(num_classes=args.num_classes)
    # elif "vgg" in args.model:
    #     model = getattr(vgg, args.model)()  # num_classes not set up
    elif "senet" == args.model:
        model = getattr(senet, "seresnet18")(class_num=args.num_classes)
    elif "densenet" == args.model:
        model = getattr(densenet, "densenet_cifar")(
            num_classes=args.num_classes)
    elif "squeezenet" in args.model:
        model = getattr(squeezenet, "squeezenet_cifar")()
    else:
        print(f"Unsupported model {args.model} or dataset {args.dataset}")
        exit(1)
    if args.dataset == "cifar10" or args.dataset == "svhn":
        print(
            f"WARNING: YOU SHOULD MAKE SURE THE MODEL {args.model} TAKES IN {args.num_classes} NUM_CLASSES FOR DATASET {args.dataset}")
    return model


def get_quantization_schedule(args):
    combination = args.apex_amp_dynamic
    if combination is None:
        return {}
    opt_cr = combination[:2]  # optimization level in critical regime
    opt_non_cr = combination[-2:]  # optimization level in non-critical regime
    ckpt_milestones = {  # hardcode the CR schedule to be 15 epochs after training starts & lr decay
        15: opt_non_cr,  # checkpoint in epoch 14, use O1 starting from epoch 15
        150: opt_cr,
        165: opt_non_cr,
        250: opt_cr,
        265: opt_non_cr
    }
    return ckpt_milestones


def get_lr(args, epoch):
    """Get the learning rate based on the config and epoch number

    Args:
        args (Namespace): Arguments
        epoch (int): 0-based index of current epoch

    Returns:
        float: Current learning rate
    """
    if args.batch_size > 128:
        # bs=512 for 300 epochs:  start from lr=0.1, scale to 0.4 linearly in 5 epochs, decay at 150 and 250
        lrs = [0.1, 0.175, 0.25, 0.325, 0.4] + 145 * \
            [0.4] + 100 * [0.04] + 50 * [0.004]
        return lrs[epoch]
    else:
        lrs = [0.1] * 150 + [0.01] * 100 + [0.001] * 50
        return lrs[epoch]  # does not scale learning rate if bs <= 128


def main():
    print("========================")
    print(f"Current time: {datetime.now().strftime('%H:%M:%S')}")
    print("Doing preparations for the training")
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    # if we don't need reproducibility across multiple executions, toggle this and get a slight performance improvement
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs_train = {'num_workers': 8,
                             'pin_memory': True,
                             'shuffle': True,
                             'sampler': None}
        cuda_kwargs_test = {'num_workers': 8,
                            'pin_memory': True,
                            'shuffle': False}
        train_kwargs.update(cuda_kwargs_train)
        test_kwargs.update(cuda_kwargs_test)

    transform_train_dict = {
        "imagenet": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        "cifar100": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ]),
        "cifar10": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        "svhn": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    }

    transform_test_dict = {
        "imagenet": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        "cifar100": transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ]),
        "cifar10": transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        "svhn": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    }

    transform_train = transform_train_dict[args.dataset]
    transform_test = transform_test_dict[args.dataset]

    default_root = './data'
    if args.dataset == "imagenet":
        train_set = datasets.ImageFolder(
            '~/data/imagenet/train',
            transform=transform_train
        )
        test_set = datasets.ImageFolder(
            '~/data/imagenet/val',
            transform=transform_test
        )
    elif args.dataset == "cifar100":
        train_set = datasets.CIFAR100(
            root=default_root,
            train=True,
            download=True,
            transform=transform_train
        )
        test_set = datasets.CIFAR100(
            root=default_root,
            train=False,
            download=True,
            transform=transform_test
        )
    elif args.dataset == "cifar10":
        train_set = datasets.CIFAR10(
            root=default_root,
            train=True,
            download=True,
            transform=transform_train
        )
        test_set = datasets.CIFAR10(
            root=default_root,
            train=False,
            download=True,
            transform=transform_test
        )
    elif args.dataset == "svhn":
        train_set = datasets.SVHN(
            root=default_root,
            split='train',
            download=True,
            transform=transform_train
        )
        test_set = datasets.SVHN(
            root=default_root,
            split='test',
            download=True,
            transform=transform_test
        )

    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)

    num_classes_dict = {
        "imagenet": 200,
        "cifar100": 100,
        "cifar10": 10,
        "svhn": 10
    }

    args.num_classes = num_classes_dict[args.dataset]

    model = get_model(args)

    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=0.0001)  # previously weight_decay=0.0001
    if args.apex_amp is not None:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.apex_amp)
    if args.apex_amp_dynamic is not None:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.apex_amp_dynamic[:2])
    scaler = torch.cuda.amp.GradScaler(enabled=args.cuda_amp)

    # scheduler = MultiStepLR(optimizer, milestones=[150, 250],
    #                         gamma=args.gamma)  # auto lr decay, schedule copied from accordion

    # set up torch.cuda.amp.autocast schedule
    autocast_schedule = []  # whether to do amp in epoch i
    if args.cuda_amp and args.cuda_amp_dynamic:
        # do full precision training in critical regimes only and amp elsewhere, copied from
        # https://github.com/uw-mad-dash/Accordion/blob/98cd7b1dd6e84cbd265dadc79421ade45e590953/main.py#L224-L248
        for i in range(300):
            if i in range(15) or i in range(150, 165) or i in range(250, 265):
                autocast_schedule.append(False)
            else:
                autocast_schedule.append(True)
    elif args.cuda_amp:  # amp throughout
        autocast_schedule = [True] * 300
    # no quantization, full precision throughout (might still do apex.amp tho)
    else:
        autocast_schedule = [False] * 300

    print("========================")
    print(f"Current time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    if args.cuda_amp:
        print(f"Precision: Auto mixed precision using torch.cuda.amp.autocast")
    elif args.apex_amp is not None:
        print(f"Precision: Using apex.amp with level {args.apex_amp}")
    elif args.apex_amp_dynamic is not None:
        print(
            f"Using apex.amp dynamically: {args.apex_amp_dynamic[:2]} at critical regime, {args.apex_amp_dynamic[-2:]} at non-critical regime")
    else:
        print(f"Precision: Default full precision")
    print(f"# epochs: {args.epochs}")
    print(f"Disable test set: {args.no_test}")
    print(f"Starting training...")
    training_start = time.time()
    print("========================")

    ckpt_milestones = get_quantization_schedule(args)

    curr_lr = args.lr  # initial learning rate

    for epoch in range(args.epochs):
        if epoch in ckpt_milestones.keys():  # will use new opt_level in current epoch
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Current epoch: {epoch}, \
                switching to opt_level {ckpt_milestones[epoch]}")
            model = get_model(args)
            model = model.to(device)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
            checkpoint = torch.load('amp_checkpoint.pt')
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=ckpt_milestones[epoch])  # init with new opt_level
            # warning: according to https://nvidia.github.io/apex/amp.html#checkpointing,
            # initializing with new opt_level is not recommended
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            amp.load_state_dict(checkpoint['amp'])
        train(args, model, device, train_loader, optimizer, epoch,
              scaler=(None if not args.cuda_amp else scaler),
              enable_amp_autocast=autocast_schedule[epoch])
        if not args.no_test:
            test(model, device, test_loader)
        # scale the learning rate if necessary
        prev_lr = curr_lr
        curr_lr = get_lr(args, epoch)
        if curr_lr != prev_lr:
            print(f"Epoch {epoch}, prev lr {prev_lr}, curr lr {curr_lr}\n")
            for group in optimizer.param_groups:
                group['lr'] = curr_lr
        if epoch+1 in ckpt_milestones.keys():  # switch to new opt_level in next epoch
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict()
            }
            torch.save(checkpoint, 'amp_checkpoint.pt')
    training_end = time.time()

    print("========================")
    print(f"Current time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    if args.cuda_amp:
        print(f"Precision: Auto mixed precision using torch.cuda.amp.autocast")
    elif args.apex_amp is not None:
        print(f"Precision: Using apex.amp with level {args.apex_amp}")
    elif args.apex_amp_dynamic is not None:
        print(
            f"Using apex.amp dynamically: {args.apex_amp_dynamic[:2]} at critical regime, {args.apex_amp_dynamic[-2:]} at non-critical regime")
    else:
        print(f"Precision: Default full precision")
    print(f"# epochs: {args.epochs}")
    print(f"Disable test set: {args.no_test}")
    print(f"JCT: {round(training_end - training_start, 2)}s")
    print("========================")


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Example')
    parser.add_argument("--model", default="resnet18", type=str,
                        help="neural net models for the training")
    parser.add_argument("--dataset", default="cifar100", type=str,
                        help="dataset for the training")
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='initial learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    # mixed/low precision training
    parser.add_argument('--cuda-amp', action='store_true', default=False,
                        help='use torch.cuda.amp.autocast() for auto mixed precision training')
    parser.add_argument('--cuda-amp-dynamic', action='store_true', default=False,
                        help='use dynamic schedule for auto mixed precision training')
    parser.add_argument('--apex-amp', type=str, default=None,
                        help='use apex.amp for custom precision training (O0/O1/O3)')
    parser.add_argument('--apex-amp-dynamic', type=str, default=None,
                        help="use different opt_level for each epoch, format: 'OA-OB' where A,B in set(0,1,3)")
    # skip test set
    parser.add_argument('--no-test', action='store_true', default=False,
                        help='skip the test set for timing purposes')
    # breakpoint for easier profiling
    parser.add_argument('--breakpoint', type=int, default=-1,
                        help='the iteration index to stop at')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
