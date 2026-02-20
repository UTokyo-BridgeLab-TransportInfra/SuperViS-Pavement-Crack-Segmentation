import os
import glob
import math
import time
import random
import argparse
from pathlib import Path
from collections import OrderedDict

import cv2
import tqdm
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from skimage import morphology
from skimage.morphology import skeletonize
from skimage.draw import disk
import sknw
import networkx as nx

from scipy.ndimage import maximum_filter, distance_transform_edt

from data_loader_augment import ImgDataSet
from unet.prf_metrics import cal_prf_metrics


# ----------------------------
# Hyperparameters
# ----------------------------
hyp = {
    "lr0": 0.005,
    "momentum": 0.98,
    "weight_decay": 4.97e-05,
    "loss_weight": 1.08,
    "crack_threshold": 0.207,
}


def fitness(x):
    """Returns fitness used by evolve logic."""
    w = [0.0, 0.2, 0.8, 0.0, 0.0]  # weights for [P, R, F1, Dice, IoU]@crack_threshold
    return (x[:, :5] * w).sum(1)


def init_seeds(seed=0):
    """Init random seeds."""
    torch.manual_seed(seed)
    if seed == 0:
        cudnn.deterministic = False
        cudnn.benchmark = True


def print_mutation(hyp_dict, results):
    """Print mutation results to evolve.txt."""
    a = "%10s" * len(hyp_dict) % tuple(hyp_dict.keys())
    b = "%10.4g" * len(hyp_dict) % tuple(hyp_dict.values())
    c = "%10.4g" * len(results) % results
    print("\n%s\n%s\nEvolved fitness: %s\n" % (a, b, c))
    with open("evolve.txt", "a") as f:
        f.write(c + b + "\n")
    x = np.unique(np.loadtxt("evolve.txt", ndmin=2), axis=0)
    np.savetxt("evolve.txt", x[np.argsort(-x[:, 3])], "%10.4g")


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        v = float(val) if not isinstance(val, float) else val
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def select_device(device="", batch_size=None):
    """
    device: 'cpu' or '0' or '0,1,2,3'
    Note: CUDA_VISIBLE_DEVICES expects standard comma separated IDs.
    """
    cpu_request = device.lower() == "cpu"
    if device and not cpu_request:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:
            assert batch_size % ng == 0, f"batch-size {batch_size} not multiple of GPU count {ng}"
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = "Using CUDA "
        for i in range(ng):
            if i == 1:
                s = " " * len(s)
            print(
                "%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)"
                % (s, i, x[i].name, x[i].total_memory / c)
            )
    else:
        print("Using CPU")

    print("")
    return torch.device("cuda:0" if cuda else "cpu")


def create_model(device):
    """Create model and load checkpoint from CLI."""
    model = smp.Unet(
        encoder_name=args.ENCODER,
        encoder_weights=args.ENCODER_WEIGHTS,
        classes=1,
        activation=None,
    )

    print(f"Loading pretrained checkpoint: {args.pretrained}")
    checkpoint = torch.load(args.pretrained, map_location=device)
    model.load_state_dict(checkpoint["model"])

    model.eval()
    return model.to(device)


def find_endpoints(skel):
    """Find endpoints of skeleton pixels (exactly one neighbor)."""
    endpoints = []
    padded = np.pad(skel.astype(int), ((1, 1), (1, 1)), mode="constant")
    for y in range(1, padded.shape[0] - 1):
        for x in range(1, padded.shape[1] - 1):
            if padded[y, x] == 1:
                neighbors = padded[y - 1 : y + 2, x - 1 : x + 2].copy()
                neighbors[1, 1] = 0
                if np.sum(neighbors) == 1:
                    endpoints.append((y - 1, x - 1))  # remove padding offset
    return np.array(endpoints)


def find_latest_model_path(dir_path):
    """Find latest 'model_epoch_*.pt' under model_dir."""
    model_paths = []
    epochs = []
    for path in Path(dir_path).glob("*.pt"):
        if "epoch" not in path.stem:
            continue
        model_paths.append(path)
        parts = path.stem.split("_")
        epoch = int(parts[-1])
        epochs.append(epoch)

    if len(epochs) > 0:
        epochs = np.array(epochs)
        max_idx = int(np.argmax(epochs))
        return model_paths[max_idx]
    return None


def remove_narrow_parts_and_small_areas(binary, width_threshold=6, min_area=50):
    """
    Crack mask cleanup:
      1) remove local narrow parts whose width < width_threshold
      2) remove small connected components with area < min_area

    Args:
        binary: uint8 mask (0/255)
        width_threshold: local width threshold (pixels)
        min_area: connected component area threshold (pixels)
    """
    bw_bool = (binary > 0)
    dist_map = distance_transform_edt(bw_bool)

    dist_limit = width_threshold / 2.0
    narrow_mask = (dist_map < dist_limit) & bw_bool
    cleaned = binary.copy()
    cleaned[narrow_mask] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    final = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            final[labels == i] = 255
    return final


def remove_small_components(binary_mask, min_area=150):
    """Remove connected components smaller than min_area. Input is 0/255 uint8."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    filtered = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255
    return filtered


def compute_grid_density(mask, grid_size=45):
    """
    Split mask into grid_size x grid_size cells and compute occupancy density.
    mask: (H, W) numpy array
    return: density in [0,1]
    """
    H, W = mask.shape
    h_step, w_step = H // grid_size, W // grid_size
    cnt, total = 0, grid_size * grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            y0 = i * h_step
            y1 = (i + 1) * h_step if i < grid_size - 1 else H
            x0 = j * w_step
            x1 = (j + 1) * w_step if j < grid_size - 1 else W
            cell = mask[y0:y1, x0:x1]
            if np.any(cell > 0):
                cnt += 1
    return cnt / total


def skeletonize_binary(mask):
    """mask: uint8 0/255 -> skeleton: uint8 0/255"""
    mask_bool = mask > 0
    skel_bool = skeletonize(mask_bool)
    return (skel_bool.astype(np.uint8)) * 255


def save_masks_with_fnames(new_target, new_targetC, fnames, save_dir, ratio_threshold=1.0):
    """
    Save masks using original filenames.
    A grid density filter is applied:
      - Compute skeleton density for new_target (A) and new_targetC (B)
      - If density_A >= ratio_threshold * density_B, skip saving A

    Args:
        new_target: torch tensor [B,1,H,W] in {0,1} float
        new_targetC: torch tensor [B,1,H,W] in {0,1} float
        fnames: list of filenames
        save_dir: output directory
        ratio_threshold: skipping ratio
    """
    os.makedirs(save_dir, exist_ok=True)
    A = new_target.detach().cpu().numpy()
    B = new_targetC.detach().cpu().numpy()

    kept = []
    for i in range(A.shape[0]):
        mask_A = (A[i, 0] * 255).astype(np.uint8)
        mask_B = (B[i, 0] * 255).astype(np.uint8)

        s_A = skeletonize_binary(mask_A)
        s_B = skeletonize_binary(mask_B)

        density_A = compute_grid_density(s_A, 25)
        density_B = compute_grid_density(s_B, 25)

        if density_A >= ratio_threshold * density_B:
            continue

        filename = fnames[i]
        cv2.imwrite(os.path.join(save_dir, filename), mask_A)
        kept.append(filename)

    return kept


def remove_short_branches_by_component(skel, length_threshold=10):
    """
    For each connected component of a skeleton (0/1), preserve the main trunk
    (graph diameter path) and prune side branches shorter than length_threshold.

    Args:
        skel: 0/1 numpy array
        length_threshold: pixels

    Returns:
        pruned skeleton (0/1)
    """
    graph = sknw.build_sknw(skel, multi=False)
    pruned_skel = skel.copy()

    for comp_nodes in nx.connected_components(graph):
        subG = graph.subgraph(comp_nodes)
        if subG.number_of_edges() == 0:
            continue

        arbitrary_node = list(subG.nodes())[0]
        distances = nx.single_source_dijkstra_path_length(subG, arbitrary_node, weight="weight")
        farthest_node = max(distances, key=distances.get)

        distances2 = nx.single_source_dijkstra_path_length(subG, farthest_node, weight="weight")
        other_end = max(distances2, key=distances2.get)

        try:
            longest_path = nx.dijkstra_path(subG, farthest_node, other_end, weight="weight")
        except Exception:
            continue

        longest_path_edges = set()
        for i in range(len(longest_path) - 1):
            longest_path_edges.add(frozenset([longest_path[i], longest_path[i + 1]]))

        for (s, e) in subG.edges():
            edge_data = subG[s][e]
            branch_length = edge_data.get("weight", 0)
            if frozenset([s, e]) in longest_path_edges:
                continue
            if branch_length < length_threshold:
                pts = np.array(edge_data["pts"], dtype=np.int32)
                pruned_skel[pts[:, 0], pts[:, 1]] = 0

    return pruned_skel


# ---------------------------------------------------------
# Missing parts in your paste: Exp + soft_dilate
# ---------------------------------------------------------
class Exp(torch.autograd.Function):
    """
    Straight-through binarization approximation used in your code:
      forward: hard threshold (0/1)
      backward: pass-through gradient
    This matches typical usage where Exp.apply(sigmoid(logits)) is used as a hard mask.
    """
    @staticmethod
    def forward(ctx, x):
        return (x > 0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def soft_dilate(x, ksize=7):
    """
    Soft dilation using max pooling (common implementation).
    Input: float tensor [B,1,H,W]
    Output: float tensor [B,1,H,W]
    """
    pad = ksize // 2
    return F.max_pool2d(x, kernel_size=ksize, stride=1, padding=pad)


# ---------------------------------------------------------
# Watershed pseudo label generation (kept logic)
# ---------------------------------------------------------
def apply_watershed_from_label_skimage(input_var, target_var, target_var1, device):
    B, C, H, W = input_var.shape
    result = torch.zeros((B, 1, H, W), dtype=torch.uint8, device=device)

    input_np = input_var.detach().cpu().numpy()
    target_np = target_var.detach().cpu().numpy()
    target_np1 = target_var1.detach().cpu().numpy()

    for i in range(B):
        img = input_np[i].transpose(1, 2, 0).astype(np.uint8)  # (H, W, 3)
        label = target_np[i, 0]   # predicted logits/prob map (H, W)
        label1 = target_np1[i, 0] # auxiliary mask (H, W)

        # Step 1: binarize
        binary = (label > 0.207).astype(np.uint8) * 255
        binaryold = (label1 > 0.207).astype(np.uint8) * 255

        # Remove narrow parts and small fragments
        binaryold = remove_narrow_parts_and_small_areas(binaryold, width_threshold=6, min_area=50)

        # Skeletonize and find endpoints
        skeleton = morphology.skeletonize(binaryold)
        endpoints = find_endpoints(skeleton)

        # Create endpoint mask to exclude dilation near endpoints
        endpoint_mask = np.zeros_like(binaryold, dtype=np.uint8)
        for y, x in endpoints:
            rr, cc = disk((y, x), 6, shape=endpoint_mask.shape)
            endpoint_mask[rr, cc] = 255

        # Dilate cracks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dilated = cv2.dilate(binaryold, kernel)

        # Remove dilated areas around endpoints
        dilated_without_endpoints = cv2.bitwise_and(dilated, cv2.bitwise_not(endpoint_mask))

        # Update binary using endpoint-excluded dilation (kept exact arithmetic style)
        binary = binary - binary * dilated_without_endpoints * 255
        binary[binary < 50] = 0
        binary[binary >= 50] = 255

        img_copy = img.copy()

        sure_fg = morphology.skeletonize(binary).astype(np.uint8)
        sure_fg = remove_short_branches_by_component(sure_fg, length_threshold=15).astype(np.uint8)
        sure_bg = binary

        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(img_copy, markers)
        markers[markers > 1] = 255
        markers[markers <= 1] = 0

        result_mask = (markers > 1).astype(np.uint8)
        result[i, 0] = torch.from_numpy(result_mask)

    return result.to(device)


# ---------------------------------------------------------
# Train / Validate (kept logic)
# ---------------------------------------------------------
def train(hyp):
    start_epoch = 0
    accumulate = max(round(64 / args.batch_size), 1)
    results_file = "results.txt"

    DIR_IMG_TRAIN = os.path.join(args.data_dir, "images", "train")
    DIR_MASK_TRAIN = os.path.join(args.data_dir, "masks", "pseudo-labels")
    DIR_MASK_TRAIN1 = os.path.join(args.data_dir, "masks", "crack_candidate_regions")
    DIR_MASK_TRAIN2 = os.path.join(args.data_dir, "masks", "reference_labels")

    DIR_IMG_VALID = os.path.join(args.data_dir, "images", "pseudo-labels")
    DIR_MASK_VALID = os.path.join(args.data_dir, "masks", "train")
    DIR_MASK_VALID1 = os.path.join(args.data_dir, "masks", "crack_candidate_regions")
    DIR_MASK_VALID2 = os.path.join(args.data_dir, "masks", "reference_labels")
    SAVE_MASK_DIR = os.path.join(args.data_dir, "masks", "pseudo-labels")


    train_img_ids = [path.name for path in Path(DIR_IMG_TRAIN).glob("*.jpg")]
    val_img_ids = [path.name for path in Path(DIR_IMG_VALID).glob("*.jpg")]
    train_mask_ids = train_img_ids
    val_mask_ids = val_img_ids
    img_names = train_img_ids + val_img_ids

    imgsz_min, imgsz_max = args.img_size, args.img_size
    imgsz_test = 448
    gs = 32
    assert math.fmod(imgsz_min, gs) == 0, f"--img-size {imgsz_min} must be a {gs}-multiple"
    args.multi_scale |= imgsz_min != imgsz_max
    if args.multi_scale:
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.1
            imgsz_max //= 0.9
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
        print("Multiscale-Training Used.")
    img_size = imgsz_max

    print(f"total images = {len(img_names)}")
    model = create_model(device)

    # Keep loss definition; ensure pos_weight on correct device (no logic change)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hyp["loss_weight"]).to(device))
    criterion_test = nn.BCEWithLogitsLoss().to(device)

    init_seeds()

    channel_means = [0.485, 0.456, 0.406]
    channel_stds = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])
    val_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])
    mask_tfms = transforms.Compose([transforms.ToTensor()])

    optimizer = optim.SGD(
        model.parameters(),
        hyp["lr0"],
        momentum=hyp["momentum"],
        weight_decay=hyp["weight_decay"],
    )

    if args.evolve:
        lf = lambda x: (((1 + math.cos(x * math.pi / 60)) / 2) ** 1.0) * 0.95 + 0.05
    else:
        lf = lambda x: (((1 + math.cos(x * math.pi / args.n_epoch)) / 2) ** 1.0) * 0.95 + 0.05
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1
    
    # Keep your DDP block as-is (even though world_size=1 is unusual)
    if device.type != "cpu" and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:9999",
            world_size=1,
            rank=0,
        )
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    rect_training = args.rect_training
    train_dataset = ImgDataSet(
        img_dir=DIR_IMG_TRAIN,
        img_size=img_size,
        img_fnames=train_img_ids,
        img_transform=train_tfms,
        mask_dir=DIR_MASK_TRAIN,
        mask_dir1=DIR_MASK_TRAIN1,
        mask_dir2=DIR_MASK_TRAIN2,
        mask_fnames=train_mask_ids,
        mask_transform=mask_tfms,
        augment=args.augment,
        hyp=hyp,
        rect=rect_training,
    )
    valid_dataset = ImgDataSet(
        img_dir=DIR_IMG_VALID,
        img_size=imgsz_test,
        img_fnames=val_img_ids,
        img_transform=val_tfms,
        mask_dir=DIR_MASK_VALID,
        mask_dir1=DIR_MASK_VALID1,
        mask_dir2=DIR_MASK_TRAIN2,
        mask_fnames=val_mask_ids,
        mask_transform=mask_tfms,
        augment=False,
        hyp=hyp,
        rect=True,
    )

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    nb = len(train_loader)

    valid_losses = []
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, F1, Dice, IoU, val_loss, train_loss


    for epoch in range(start_epoch, args.n_epoch):

        print(optimizer.param_groups[0]["lr"])
    
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description(f"Epoch {epoch}")
        losses = AverageMeter()
    
        model.train()
        for i, (input, target, target1, target2, fnames) in enumerate(train_loader):
            ni = i + nb * epoch
    
            input_var = input.to(device)
            pseudo_label = target.to(device)
            crack_candidate_region = target2.to(device)
            reference_ground_truth_clear = target1.to(device)
    
            if args.multi_scale:
                if ni / accumulate % 1 == 0:
                    img_size = random.randrange(grid_min, grid_max + 1) * gs
                sf = img_size / max(input_var.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in input_var.shape[2:]]
                    input_var = F.interpolate(input_var, size=ns, mode="bilinear", align_corners=False)
                    pseudo_label = torch.unsqueeze(pseudo_label, 1)
                    pseudo_label = F.interpolate(pseudo_label, size=ns, mode="bilinear", align_corners=False)
    
            masks_preds = model(input_var)
            masks_pred = Exp.apply(torch.sigmoid(masks_preds))
    
            with torch.no_grad():
    
                # 1) Threshold pseudo-label
                pseudo_label = (pseudo_label > 0.1).float()
    
                # 2) reshape to [B,1,448,448]
                pseudo_label_reshaped = pseudo_label.view(pseudo_label.size(0), 1, 448, 448)
                reference_ground_truth_clear = reference_ground_truth_clear.view(
                    reference_ground_truth_clear.size(0), 1, 448, 448
                )
                crack_candidate_region = crack_candidate_region.view(
                    crack_candidate_region.size(0), 1, 448, 448
                )
    
                # 3) watershed refinement
                new_target = apply_watershed_from_label_skimage(
                    input_var, masks_preds, pseudo_label_reshaped, device
                )
    
                # 4) apply clear-image foreground mask
                new_target = new_target * reference_ground_truth_clear
    
                # 5) merge pseudo + refined
                new_target = (pseudo_label_reshaped + new_target).clamp(0, 1)
    
                # 6) convert to numpy
                binary = (new_target > 0.5).cpu().numpy()
    
                filtered_targets = np.zeros_like(binary, dtype=bool)
    
                for bi in range(binary.shape[0]):
                    bin_img = binary[bi, 0]
    
                    distance = distance_transform_edt(bin_img)
                    local_max_distance = maximum_filter(distance, size=20)
                    local_width_map = local_max_distance * 2
    
                    mask_to_remove = (local_width_map > 20) & bin_img
                    filtered = bin_img & (~mask_to_remove)
                    filtered = morphology.remove_small_objects(filtered.astype(bool), min_size=50)
    
                    filtered_targets[bi, 0] = filtered
    
                new_target = torch.from_numpy(filtered_targets.astype(np.float32)).to(device)
    
                # save pseudo labels
                save_masks_with_fnames(
                    new_target,
                    crack_candidate_region,
                    fnames,
                    save_dir=SAVE_MASK_DIR,
                )
    
            # loss (unchanged)
            loss = 0.5 * criterion(masks_preds, pseudo_label_reshaped) + \
                   0.5 * criterion(masks_preds, new_target)
    
            loss *= args.batch_size / 64
            losses.update(loss)
    
            tq.set_postfix(OrderedDict(train_loss="{:.5f}".format(losses.avg), size=img_size))
            tq.update(args.batch_size)
    
            loss.backward()
    
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
    
        scheduler.step()
        tq.close()


        final_epoch = (epoch + 1 == args.n_epoch)
        if not args.notest or final_epoch:
            valid_metrics = validate(model, valid_loader, criterion_test, epoch, hyp["crack_threshold"])
            valid_loss = valid_metrics["valid_loss"]
            unet_precision = valid_metrics["precision"]
            unet_recall = valid_metrics["recall"]
            unet_F1 = valid_metrics["F1"]
            unet_dice = valid_metrics["mean_dice"]
            unet_IOU = valid_metrics["mean_IOU"]
            valid_losses.append(valid_loss)
            results = (unet_precision, unet_recall, unet_F1, unet_dice, unet_IOU, valid_loss, losses.avg)
            results1 = (unet_precision, unet_recall, unet_F1, unet_dice, unet_IOU)

        with open(results_file, "a") as f:
            f.write("%10.5g" * 7 % results + "\n")

    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results


def validate(model, val_loader, criterion_test, epoch, crack_threshold):
    SAVE_MASK_DIR = os.path.join(args.data_dir, "masks", "train")
    losses = AverageMeter()
    model.eval()

    pred = np.array([]).reshape(0, 448, 448)
    targ = np.array([]).reshape(0, 448, 448)

    tq = tqdm.tqdm(total=(len(val_loader) * args.batch_size))
    tq.set_description(f"Epoch {epoch}")

    # metric collectors
    precision_list = []
    recall_list = []
    f1_list = []
    dice_list = []
    iou_list = []

    with torch.no_grad():
        for _, (input, target, target1, target2, fnames) in enumerate(val_loader):

            input_var = input.to(device)

            # renamed labels
            pseudo_label = target.unsqueeze(1).to(device)
            reference_ground_truth_clear = target1.unsqueeze(1).to(device)
            crack_candidate_region = target2.view(target2.size(0), 1, 448, 448).to(device)

            pseudo_label_reshaped = pseudo_label.view(pseudo_label.size(0), 1, 448, 448)

            # forward
            output = model(input_var)

            # watershed refinement
            new_target = apply_watershed_from_label_skimage(
                input_var, output, pseudo_label_reshaped, device
            )

            reference_ground_truth_clear = reference_ground_truth_clear.view(
                reference_ground_truth_clear.size(0), 1, 448, 448
            )

            new_target = new_target * reference_ground_truth_clear
            new_target = (pseudo_label_reshaped + new_target).clamp(0, 1)

            # ---- width filtering ----
            binary = (new_target > 0.5).cpu().numpy()
            filtered_targets = np.zeros_like(binary, dtype=bool)

            for bi in range(binary.shape[0]):
                bin_img = binary[bi, 0]
                distance = distance_transform_edt(bin_img)
                local_max_distance = maximum_filter(distance, size=20)
                local_width_map = local_max_distance * 2
                mask_to_remove = (local_width_map > 20) & bin_img
                filtered = bin_img & (~mask_to_remove)
                filtered = morphology.remove_small_objects(filtered.astype(bool), min_size=50)
                filtered_targets[bi, 0] = filtered

            new_target = torch.from_numpy(filtered_targets.astype(np.float32)).to(device)

            # save pseudo labels
            save_masks_with_fnames(
                new_target,
                crack_candidate_region,
                fnames,
                save_dir=SAVE_MASK_DIR,
            )

            # validation loss
            loss = criterion_test(output, pseudo_label)
            losses.update(loss.item(), input_var.size(0))

            tq.set_postfix(valid_loss="{:.5f}".format(losses.avg))
            tq.update(args.batch_size)

            # collect predictions
            pred_1 = np.concatenate(
                (pred, torch.sigmoid(output).squeeze(1).cpu().numpy()), axis=0
            )
            targ_1 = np.concatenate(
                (targ, pseudo_label.squeeze(1).cpu().numpy()), axis=0
            )

            # compute metrics
            p_acc, r_acc, F1, mean_dice, mean_jaccard = cal_prf_metrics(
                pred_1, targ_1, crack_threshold
            )

            precision_list.append(p_acc)
            recall_list.append(r_acc)
            f1_list.append(F1)
            dice_list.append(mean_dice)
            iou_list.append(mean_jaccard)

    # ---- epoch metrics ----
    epoch_precision = np.nanmean(np.array(precision_list))
    epoch_recall    = np.nanmean(np.array(recall_list))
    epoch_f1        = np.nanmean(np.array(f1_list))
    epoch_dice      = np.nanmean(np.array(dice_list))
    epoch_iou       = np.nanmean(np.array(iou_list))

    print(epoch_recall)
    tq.close()

    return {
        "valid_loss": losses.avg,
        "precision": epoch_precision,
        "recall": epoch_recall,
        "F1": epoch_f1,
        "mean_dice": epoch_dice,
        "mean_IOU": epoch_iou,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Training Script")

    parser.add_argument("-n_epoch", default=10, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-print_freq", default=30, type=int, metavar="N", help="print frequency")
    parser.add_argument("-img_size", nargs="+", type=int, default=448, help="[min_train, max-train, test]")
    parser.add_argument("-batch_size", default=4, type=int, help="batch size")
    parser.add_argument("-num_workers", default=16, type=int, help="num dataloader workers")
    parser.add_argument("-device", default="1", help="device id (e.g. 0 or 0,1 or cpu)")

    parser.add_argument("-data_dir", type=str, required=True, help="input dataset directory")
    parser.add_argument("-model_dir", type=str, required=False, help="output model directory")

    parser.add_argument("-evolve", action="store_true", help="evolve hyperparameters")
    parser.add_argument("-ENCODER", type=str, required=False, default="vgg16")
    parser.add_argument("-ENCODER_WEIGHTS", default="imagenet")

    parser.add_argument("-augment", action="store_true", help="use data augmentation")
    parser.add_argument("-nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("-notest", action="store_true", help="only test final epoch")
    parser.add_argument("-multi-scale", dest="multi_scale", action="store_true",
                        help="adjust image size periodically during training")
    parser.add_argument("-rect_training", action="store_true", help="rectangular training mode")
    parser.add_argument("-pretrained", type=str, required=True, help="path to pretrained checkpoint")


    args = parser.parse_args()

    device = select_device(device=args.device, batch_size=args.batch_size)

    
    train(hyp)
    
