import numpy as np

def cal_semantic_metrics(pred_list, gt_list, thresh=0.2, num_cls=2):
    global_accuracy_cur = []
    statistics = []
    result_dice = []
    result_jaccard = []
    for pred, gt in zip(pred_list, gt_list):
        gt_img   = gt.astype('uint8')
        pred_img = (pred > thresh).astype('uint8')
            # calculate each image
        global_accuracy_cur.append(cal_global_acc(pred_img, gt_img))
        statistics.append(get_statistics(pred_img, gt_img, num_cls))
        result_dice += [dice(gt_img, pred_img)]
        result_jaccard += [jaccard(gt_img, pred_img)]        
        # get global accuracy with corresponding threshold: (TP+TN)/all_pixels
    global_acc = np.sum([v[0] for v in global_accuracy_cur])/np.sum([v[1] for v in global_accuracy_cur])
    mean_dice = np.mean(result_dice)
    mean_jaccard= np.mean(result_jaccard)
        # get tp, fp, fn
    counts = []
    for i in range(num_cls):
        tp = np.sum([v[i][0] for v in statistics])
        fp = np.sum([v[i][1] for v in statistics])
        fn = np.sum([v[i][2] for v in statistics])
        counts.append([tp, fp, fn])

        # calculate mean accuracy
    mean_acc = np.sum([v[0]/(v[0]+v[2]) for v in counts])/num_cls
        # calculate mean iou
    mean_iou_acc = np.sum([v[0]/(np.sum(v)) for v in counts])/num_cls
    final_accuracy_all = [global_acc, mean_acc, mean_iou_acc, mean_dice, mean_jaccard]

    return final_accuracy_all

def cal_global_acc(pred, gt):
    """
    acc = (TP+TN)/all_pixels
    """
    h,w = gt.shape
    return [np.sum(pred==gt), float(h*w)]

def get_statistics(pred, gt, num_cls=2):
    """
    return tp, fp, fn
    """
    h,w = gt.shape
    statistics = []
    for i in range(num_cls):
        tp = np.sum((pred==i)&(gt==i))
        fp = np.sum((pred==i)&(gt!=i))
        fn = np.sum((pred!=i)&(gt==i)) 
        statistics.append([tp, fp, fn])
    return statistics

def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)