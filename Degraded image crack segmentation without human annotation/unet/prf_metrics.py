import numpy as np

def cal_prf_metrics(pred_list, gt_list, thresh=0.5):

    statistics = []
    result_dice = []
    result_jaccard = []        
    for pred, gt in zip(pred_list, gt_list):
        gt_img   = gt.astype('uint8')
        pred_img = (pred > thresh).astype('uint8')
        # calculate each image
        statistics.append(get_statistics(pred_img, gt_img))
        result_dice += [dice(gt_img, pred_img)]
        result_jaccard += [jaccard(gt_img, pred_img)]         
    # get tp, fp, fn
    mean_dice = np.mean(result_dice)
    mean_jaccard= np.mean(result_jaccard)
    tp = np.sum([v[0] for v in statistics])
    fp = np.sum([v[1] for v in statistics])
    fn = np.sum([v[2] for v in statistics])

    # calculate precision
    p_acc = 1.0 if tp==0 and fp==0 else tp/(tp+fp)
    # calculate recall
    r_acc = tp/(tp+fn)
    # calculate f-score
    final_accuracy_all = [p_acc, r_acc, 2*p_acc*r_acc/(p_acc+r_acc), mean_dice, mean_jaccard]
    return final_accuracy_all

def get_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn] 

def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)