import torch
from torch import nn
from unet.unet_transfer import UNet16, UNetResNet
from unet.prf_metrics import cal_prf_metrics
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.autograd import Variable
import shutil
from data_loader_augment import ImgDataSet
import os
import cv2
from collections import OrderedDict
import argparse
import tqdm
import numpy as np
import scipy.ndimage as ndimage
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import train_test_split
import math
import random
import time
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import glob
from utils import plot_lr_scheduler
import numpy as np
import segmentation_models_pytorch as smp

hyp = {'lr0': 0.0134,
        'momentum': 0.98,
        'weight_decay': 4.97e-05,
        'loss_weight': 1.08,
        'crack_threshold': 0.207}


class BinaryFocalLoss(nn.Module):    # used in retinanet, for balancing positive and negative samples.
    # previous alpha = 1, gamma = 2  in the paper alpha = 0.25, gamma = 2
    def __init__(self, alpha=1, gamma=2, logits=False, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.size_average:
            return F_loss.mean()
        else:
            return F_loss.sum()

def fitness(x):
    # Returns fitness (for use with results.txt or evolve.txt)
    w = [0.0, 0.2, 0.0, 0.8, 0.0]  # weights for [P, R, F1, Dice, IoU]@crack_threshold
    return (x[:, :5] * w).sum(1)

def init_seeds(seed=0):
    torch.manual_seed(seed)
    # Reduce randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        cudnn.deterministic = False
        cudnn.benchmark = True

def print_mutation(hyp, results):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.4g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.4g' * len(results) % results
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))
    with open('evolve.txt', 'a') as f: 
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)
    np.savetxt('evolve.txt', x[np.argsort(-x[:,3])], '%10.4g')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity
    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA '  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')

def create_model(device):
    model = smp.Unet(
        encoder_name=args.ENCODER, 
        encoder_weights=args.ENCODER_WEIGHTS, 
        classes=1, 
        activation=None,
    )
    model.eval()
    return model.to(device)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def find_latest_model_path(dir):
    model_paths = []
    epochs = []
    for path in Path(dir).glob('*.pt'):
        if 'epoch' not in path.stem:
            continue
        model_paths.append(path)
        parts = path.stem.split('_')
        epoch = int(parts[-1])
        epochs.append(epoch)

    if len(epochs) > 0:
        epochs = np.array(epochs)
        max_idx = np.argmax(epochs)
        return model_paths[max_idx]
    else:
        return None

def train(hyp):
    start_epoch = 0
    accumulate = max(round(64 / args.batch_size), 1)
    results_file = 'results.txt'
    DIR_IMG_TRAIN  = os.path.join(args.data_dir, 'images', 'train')
    DIR_MASK_TRAIN = os.path.join(args.data_dir, 'masks', 'train')
    DIR_IMG_VALID  = os.path.join(args.data_dir, 'images', 'valid')
    DIR_MASK_VALID = os.path.join(args.data_dir, 'masks', 'valid')
    train_img_ids = [path.name for path in Path(DIR_IMG_TRAIN).glob('*.jpg')]
    val_img_ids = [path.name for path in Path(DIR_IMG_VALID).glob('*.jpg')]
    train_mask_ids = train_img_ids
    val_mask_ids = val_img_ids
    img_names  = train_img_ids + val_img_ids

    imgsz_min, imgsz_max = args.img_size, args.img_size  # img sizes (min, max, test)
    imgsz_test =448
    gs = 32
    assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
    args.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
    if args.multi_scale:
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.1
            imgsz_max //= 0.9
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
        print('Multiscale-Training Used.')
    img_size = imgsz_max  # initialize with max size    
    print(f'total images = {len(img_names)}')
    model = create_model(device)
    
    FocalLoss = False           # loss mod or FocalLoss mod
    if FocalLoss:
        criterion = BinaryFocalLoss(gamma = hyp['fl_gamma']).to(device)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hyp['loss_weight'])).to(device)
#         criterion = nn.BCEWithLogitsLoss().to(device)
    criterion_test = nn.BCEWithLogitsLoss().to(device)
    init_seeds()
    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(channel_means, channel_stds)])

    val_tfms = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(channel_means, channel_stds)])

    optimizer = torch.optim.SGD(model.parameters(), hyp['lr0'],
                                momentum=hyp['momentum'],
                                weight_decay=hyp['weight_decay'])
    if args.evolve:
        lf = lambda x: (((1 + math.cos(x * math.pi / 60)) / 2) ** 1.0) * 0.95 + 0.05
    else:  
        lf = lambda x: (((1 + math.cos(x * math.pi / args.n_epoch)) / 2) ** 1.0) * 0.95 + 0.05
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1
    
    mask_tfms = transforms.Compose([transforms.ToTensor()])    
    latest_model_path = find_latest_model_path(args.model_dir)
    best_model_path = os.path.join(*[args.model_dir, 'model_best.pt'])

    start_epoch = 0
    max_iou = 0.0
    if latest_model_path is not None and not args.evolve:
        state = torch.load(latest_model_path, map_location=device)
        start_epoch = state['epoch']
        model.load_state_dict(state['model'])
        if state['optimizer'] is not None:
            optimizer.load_state_dict(state['optimizer'])
            max_iou = state['max_iou']
        start_epoch = state['epoch'] + 1

        #if latest model path does exist, best_model_path should exists as well
        assert Path(best_model_path).exists() == True, f'best model path {best_model_path} does not exist'
        #load the min loss so far
        best_state = torch.load(best_model_path, map_location=device)
        min_train_los = best_state['train_loss']

        print(f'Restored model at epoch {start_epoch}. Min training loss so far is : {min_train_los}')
        print(f'Started training model from epoch {start_epoch}')
    else:
        print('Started training model from epoch 0')
        for f in glob.glob(results_file):
            os.remove(f)
#     plot_lr_scheduler(optimizer, scheduler, epochs = args.n_epoch)
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    
    # define the train loader and validation loader
#     augment = True
    rect_training = args.rect_training
    train_dataset = ImgDataSet(img_dir=DIR_IMG_TRAIN,img_size=img_size, img_fnames=train_img_ids, img_transform=train_tfms, mask_dir=DIR_MASK_TRAIN, mask_fnames=train_mask_ids, mask_transform=mask_tfms, augment = args.augment, hyp = hyp, rect = rect_training)
    valid_dataset = ImgDataSet(img_dir=DIR_IMG_VALID,img_size=imgsz_test, img_fnames=val_img_ids, img_transform=val_tfms, mask_dir=DIR_MASK_VALID, mask_fnames=val_mask_ids, mask_transform=mask_tfms, augment = False, hyp = hyp, rect = True)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
    nb = len(train_loader)
    
    valid_losses = []
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'F1', 'Dice', 'IoU', 'validate loss', 'train loss'
    for epoch in range(start_epoch, args.n_epoch):
        print(optimizer.param_groups[0]['lr'])
#         adjust_learning_rate(optimizer, epoch, args.lr)
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}')
        losses = AverageMeter()

        model.train()
        for i, (input, target) in enumerate(train_loader):
            ni = i + nb*epoch
            input_var = input.to(device)
            target_var = target.to(device)
            if args.multi_scale:
                if ni /accumulate % 1 == 0:
                    img_size = random.randrange(grid_min, grid_max + 1) * gs
                sf = img_size / max(input_var.shape[2:])  # scale factor    
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in input_var.shape[2:]]
                    input_var = F.interpolate(input_var, size=ns, mode='bilinear', align_corners=False)
                    target_var = torch.unsqueeze(target_var, 1)
                    target_var= F.interpolate(target_var, size=ns, mode='bilinear', align_corners=False)
            masks_pred = model(input_var)
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat  = target_var.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            loss *= args.batch_size / 64
            losses.update(loss)
#             
            tq.set_postfix(OrderedDict(train_loss='{:.5f}'.format(losses.avg), size = img_size))
            tq.update(args.batch_size)
            loss.backward()
           
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
            
        scheduler.step()
        tq.close()
        final_epoch = epoch + 1 == args.n_epoch
        if not args.notest or final_epoch:
            valid_metrics = validate(model, valid_loader, criterion_test, epoch, hyp['crack_threshold'])
            valid_loss = valid_metrics['valid_loss']
            unet_precision = valid_metrics['precision']
            unet_recall = valid_metrics['recall']
            unet_F1 = valid_metrics['F1']
            unet_dice = valid_metrics['mean_dice']
            unet_IOU = valid_metrics['mean_IOU']
            valid_losses.append(valid_loss)
            results = (unet_precision, unet_recall, unet_F1, unet_dice, unet_IOU, valid_loss, losses.avg)
            results1 = (unet_precision, unet_recall, unet_F1, unet_dice, unet_IOU)
        with open(results_file, 'a') as f:
            f.write('%10.5g' * 7 % results + '\n')  # P, R, F1, dice, iou      
        fi = fitness(np.array(results1).reshape(1, -1))
        #save the model of the current epoch
        if not args.evolve:
            epoch_model_path = os.path.join(*[args.model_dir, f'model_epoch_{epoch}.pt'])
            torch.save({
                'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'epoch': epoch,
                'max_iou': fi,
                'train_loss': losses.avg,
                'optimizer': None if final_epoch else optimizer.state_dict()
            }, epoch_model_path)

            if fi > max_iou:
                max_iou = fi

                torch.save({
                    'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'epoch': epoch,
                    'max_iou': fi,
                    'train_loss': losses.avg,
                    'optimizer': None if final_epoch else optimizer.state_dict()
                }, best_model_path)
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results

def validate(model, val_loader, criterion_test, epoch, crack_threshold):
    losses = AverageMeter()
    model.eval()
    pred = np.array([]).reshape(0,448,448)
    targ =np.array([]).reshape(0,448,448)
    tq = tqdm.tqdm(total=(len(val_loader) * args.batch_size))
    tq.set_description(f'Epoch {epoch}')
    p_acc1=[]
    r_acc1=[]
    F11=[]
    mean_dice1=[]
    mean_jaccard1=[]
    
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var = input.to(device)
            target_var = target.unsqueeze(1).to(device)       
            output = model(input_var)
           
            
            loss = criterion_test(output, target_var)
            losses.update(loss.item(), input_var.size(0))
            tq.set_postfix(valid_loss='{:.5f}'.format(losses.avg))
            tq.update(args.batch_size)
            pred22 = np.concatenate((pred,torch.sigmoid(output).squeeze(1).cpu().numpy()),axis=0)
            targ22 = np.concatenate((targ,target_var.squeeze(1).cpu().numpy()),axis=0)
            #targ = numpy.(target_var.squeeze(1).cpu().numpy())
            
            [p_acc, r_acc, F1, mean_dice, mean_jaccard] = cal_prf_metrics(pred22, targ22, crack_threshold)
            #print(p_acc)
            
            p_acc1=np.append(p_acc1,p_acc)
            r_acc1=np.append(r_acc1,r_acc)
            F11=np.append(F11,F1)
            mean_dice1=np.append(mean_dice1,mean_dice)
            mean_jaccard1=np.append(mean_jaccard1,mean_jaccard)
    p_acck=np.nanmean(p_acc1)
    r_acck=np.nanmean(r_acc1)
    F1k=np.nanmean(F11)
    mean_dicek=np.nanmean(mean_dice1)
    mean_jaccardk=np.nanmean(mean_jaccard1)
  
    tq.close()
    return {'valid_loss': losses.avg, 'precision': p_acck, 'recall': r_acck, 'F1': F1k, 'mean_dice':mean_dicek,'mean_IOU': mean_jaccardk}

def save_check_point(state, is_best, file_name = 'checkpoint.pth.tar'):
    torch.save(state, file_name)
    if is_best:
        shutil.copy(file_name, 'model_best.pth.tar')

def calc_crack_pixel_weight(mask_dir):
    avg_w = 0.0
    n_files = 0
    for path in Path(mask_dir).glob('*.*'):
        n_files += 1
        m = ndimage.imread(path)
        ncrack = np.sum((m > 0)[:])
        w = float(ncrack)/(m.shape[0]*m.shape[1])
        avg_w = avg_w + (1-w)

    avg_w /= float(n_files)

    return avg_w / (1.0 - avg_w)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-n_epoch', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-print_freq', default=30, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-img_size', nargs='+', type=int, default=448, help='[min_train, max-train, test]')
    parser.add_argument('-batch_size',  default=4, type=int,  help='weight decay (default: 1e-4)')
    parser.add_argument('-num_workers', default=4, type=int, help='output dataset directory')
    parser.add_argument('-device',default= '1', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('-data_dir',type=str, help='input dataset directory')
    parser.add_argument('-model_dir', type=str, help='output dataset directory')
    parser.add_argument('-evolve',  action='store_true', help='evolve hyperparameters')
    parser.add_argument('-ENCODER', type=str, required=False, default='vgg16')
    parser.add_argument('-ENCODER_WEIGHTS', default= 'imagenet')
    parser.add_argument('-augment', action='store_true', help='using data augmentation')
    parser.add_argument('-nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('-notest', action='store_true', help='only test final epoch')
    parser.add_argument('-multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('-rect_training', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')

    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    device = select_device(device = args.device, batch_size = args.batch_size)
    if not args.evolve: # Train normally
        train(hyp)
    else: # Evolve hyperparameters (optional)
        args.notest, args.nosave = True, True
        for _ in range(100):
            print('GA parameter searching:', _)
            if os.path.exists('evolve.txt'):
                parent = 'single'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))
                x = x[np.argsort(-fitness(x))][:n]
                w = fitness(x) - fitness(x).min()
                if parent == 'single' or len(x) == 1:
                    x = x[random.choices(range(n), weights=w)[0]]
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()
                method, mp, s = 3, 0.9, 0.2
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([1, .1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 0, 1])
                ng = len(g)
                if method == 1:
                    v = (npr.randn(ng) * npr.random() * g * s + 1) ** 2.0
                elif method == 2:
                    v = (npr.randn(ng) * npr.random(ng) * g * s + 1) ** 2.0
                elif method == 3:
                    v = np.ones(ng)
                    while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                        # v = (g * (npr.random(ng) < mp) * npr.randn(ng) * s + 1) ** 2.0
                        v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = x[i + 7] * v[i]  # mutate

            # Clip to limits
            keys = ['lr0', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'hsv_h','degrees', 'translate', 'scale', 'shear', 'loss_weight', 'fl_gamma', 'crack_threshold', 'perspective']
            limits = [(1e-5, 3e-2),  (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .2), (0, 45), (0, .9), (0, .9), (0.0, 10.0), (0.1, 50), (0, 3), (0.1, 0.5), (0, 0.002)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            results = train(hyp.copy())
            # Write mutation results
            print_mutation(hyp, results)
