import numpy as np
import nibabel as nib
import cc3d
import scipy
import os
import pandas as pd
import surface_distance
import sys
import math
from medpy.metric.binary import __surface_distances


def normalized_surface_dice(a: np.ndarray, b: np.ndarray, threshold: float, spacing: tuple = None, connectivity=1):
    """
    Pulled from nnUNet - 
    https://github.com/MIC-DKFZ/nnUNet/blob/4f2ffabe751977ee66348560c8e99102e8553195/nnunet/evaluation/surface_dice.py#L16

    This implementation differs from the official surface dice implementation! These two are not comparable!!!!!

    The normalized surface dice is symmetric, so it should not matter whether a or b is the reference image

    This implementation natively supports 2D and 3D images. Whether other dimensions are supported depends on the
    __surface_distances implementation in medpy

    :param a: image 1, must have the same shape as b
    :param b: image 2, must have the same shape as a
    :param threshold: distances below this threshold will be counted as true positives. Threshold is in mm, not voxels!
    (if spacing = (1, 1(, 1)) then one voxel=1mm so the threshold is effectively in voxels)
    must be a tuple of len dimension(a)
    :param spacing: how many mm is one voxel in reality? Can be left at None, we then assume an isotropic spacing of 1mm
    :param connectivity: see scipy.ndimage.generate_binary_structure for more information. I suggest you leave that
    one alone
    :return:
    """
    assert all([i == j for i, j in zip(a.shape, b.shape)]), "a and b must have the same shape. a.shape= %s, " \
                                                            "b.shape= %s" % (
                                                                str(a.shape), str(b.shape))
    if spacing is None:
        spacing = tuple([1 for _ in range(len(a.shape))])
    a_to_b = __surface_distances(a, b, spacing, connectivity)
    b_to_a = __surface_distances(b, a, spacing, connectivity)

    numel_a = len(a_to_b)
    numel_b = len(b_to_a)

    tp_a = np.sum(a_to_b <= threshold) / numel_a
    tp_b = np.sum(b_to_a <= threshold) / numel_b

    fp = np.sum(a_to_b > threshold) / numel_a
    fn = np.sum(b_to_a > threshold) / numel_b

    # 1e-8 just so that we don't get div by 0
    dc = (tp_a + tp_b) / (tp_a + tp_b + fp + fn + 1e-8)
    return dc


def dice(im1, im2):
    """
    Computes Dice score for two images

    Parameters
    ==========
    im1: Numpy Array/Matrix; Predicted segmentation in matrix form 
    im2: Numpy Array/Matrix; Ground truth segmentation in matrix form

    Output
    ======
    dice_score: Dice score between two images
    """

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError(
            "Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * (intersection.sum()) / (im1.sum() + im2.sum())


def get_TissueWiseSeg(prediction_matrix, gt_matrix, tissue_type):
    """
    Converts the segmentatations to isolate tissue types

    Parameters
    ==========
    prediction_matrix: Numpy Array/Matrix; Predicted segmentation in matrix form 
    gt_matrix: Numpy Array/Matrix; Ground truth segmentation in matrix form
    tissue_type: str; Can be WT, ET or TC

    Output
    ======
    prediction_matrix: Numpy Array/Matrix; Predicted segmentation in matrix form with 
                       just tissue type mentioned
    gt_matrix: Numpy Array/Matrix; Ground truth segmentation in matrix form with just 
                       tissue type mentioned
    """

    if tissue_type == 'WT':
        np.place(prediction_matrix, (prediction_matrix != 1) & (
            prediction_matrix != 2) & (prediction_matrix != 3), 0)
        np.place(prediction_matrix, (prediction_matrix > 0), 1)

        np.place(gt_matrix, (gt_matrix != 1) & (
            gt_matrix != 2) & (gt_matrix != 3), 0)
        np.place(gt_matrix, (gt_matrix > 0), 1)

    elif tissue_type == 'TC':
        np.place(prediction_matrix, (prediction_matrix != 1)
                 & (prediction_matrix != 3), 0)
        np.place(prediction_matrix, (prediction_matrix > 0), 1)

        np.place(gt_matrix, (gt_matrix != 1) & (gt_matrix != 3), 0)
        np.place(gt_matrix, (gt_matrix > 0), 1)

    elif tissue_type == 'ET':
        np.place(prediction_matrix, (prediction_matrix != 3), 0)
        np.place(prediction_matrix, (prediction_matrix > 0), 1)

        np.place(gt_matrix, (gt_matrix != 3), 0)
        np.place(gt_matrix, (gt_matrix > 0), 1)

    return prediction_matrix, gt_matrix


def get_GTseg_combinedByDilation(gt_dilated_cc_mat, gt_label_cc):
    """
    Computes the Corrected Connected Components after combing lesions
    together with respect to their dilation extent

    Parameters
    ==========
    gt_dilated_cc_mat: Numpy Array/Matrix; Ground Truth Dilated Segmentation 
                       after CC Analysis
    gt_label_cc: Numpy Array/Matrix; Ground Truth Segmentation after 
                       CC Analysis

    Output
    ======
    gt_seg_combinedByDilation_mat: Numpy Array/Matrix; Ground Truth 
                                   Segmentation after CC Analysis and 
                                   combining lesions
    """

    gt_seg_combinedByDilation_mat = np.zeros_like(gt_dilated_cc_mat)

    for comp in range(np.max(gt_dilated_cc_mat)):
        comp += 1

        gt_d_tmp = np.zeros_like(gt_dilated_cc_mat)
        gt_d_tmp[gt_dilated_cc_mat == comp] = 1
        gt_d_tmp = (gt_label_cc*gt_d_tmp)

        np.place(gt_d_tmp, gt_d_tmp > 0, comp)
        gt_seg_combinedByDilation_mat += gt_d_tmp

    return gt_seg_combinedByDilation_mat


def get_LesionWiseScores(prediction_seg, gt_seg, label_value, dil_factor):
    """
    Computes the Lesion-wise scores for pair of prediction and ground truth
    segmentations

    Parameters
    ==========
    prediction_seg: str; location of the prediction segmentation    
    gt_label_cc: str; location of the gt segmentation
    label_value: str; Can be WT, ET or TC
    dil_factor: int; Used to perform dilation

    Output
    ======
    tp: Number of TP lesions WRT prediction segmentation
    fn: Number of FN lesions WRT prediction segmentation
    fp: Number of FP lesions WRT prediction segmentation 
    gt_tp: Number of Ground Truth TP lesions WRT prediction segmentation 
    metric_pairs: list; All the lesion-wise metrics  
    full_dice: Dice Score of the pair of segmentations
    full_gt_vol: Total Ground Truth Segmenatation Volume
    full_pred_vol: Total Prediction Segmentation Volume
    """

    # Get Prediction and GT segs matrix files
    pred_nii = nib.load(prediction_seg)
    gt_nii = nib.load(gt_seg)
    pred_mat = pred_nii.get_fdata()
    gt_mat = gt_nii.get_fdata()

    # Get Spacing to computes volumes
    # Brats Assumes all spacing is 1x1x1mm3
    sx, sy, sz = pred_nii.header.get_zooms()

    # Get the prediction and GT matrix based on
    # WT, TC, ET

    pred_mat, gt_mat = get_TissueWiseSeg(
        prediction_matrix=pred_mat,
        gt_matrix=gt_mat,
        tissue_type=label_value
    )

    # Get Dice score for the full image
    if np.all(gt_mat == 0) and np.all(pred_mat == 0):
        full_dice = 1.0
    else:
        full_dice = dice(
            pred_mat,
            gt_mat
        )

    if np.any(gt_mat > 0) and np.all(pred_mat == 0):
        full_nsd_05 = 0
        full_nsd_10 = 0
    elif np.any(pred_mat > 0) and np.all(gt_mat == 0):
        full_nsd_05 = 0
        full_nsd_10 = 0
    elif np.all(pred_mat == 0) and np.all(gt_mat == 0):
        full_nsd_05 = 1
        full_nsd_10 = 1
    else:
        full_nsd_05 = normalized_surface_dice(
            pred_mat,
            gt_mat,
            threshold=0.5,
            spacing=(sx, sy, sz),
            connectivity=1
        )
        full_nsd_10 = normalized_surface_dice(
            pred_mat,
            gt_mat,
            threshold=1.0,
            spacing=(sx, sy, sz),
            connectivity=1
        )

    # Get HD95 sccre for the full image

    if np.all(gt_mat == 0) and np.all(pred_mat == 0):
        full_hd95 = 0.0
    else:
        full_sd = surface_distance.compute_surface_distances(gt_mat.astype(int),
                                                             pred_mat.astype(
                                                                 int),
                                                             (sx, sy, sz))
        full_hd95 = surface_distance.compute_robust_hausdorff(full_sd, 95)

    # Get Sensitivity and Specificity
    full_sens, full_specs = get_sensitivity_and_specificity(result_array=pred_mat,
                                                            target_array=gt_mat)

    # Get GT Volume and Pred Volume for the full image
    full_gt_vol = np.sum(gt_mat)*sx*sy*sz
    full_pred_vol = np.sum(pred_mat)*sx*sy*sz

    # Performing Dilation and CC analysis

    dilation_struct = scipy.ndimage.generate_binary_structure(3, 2)

    gt_mat_cc = cc3d.connected_components(gt_mat, connectivity=26)
    pred_mat_cc = cc3d.connected_components(pred_mat, connectivity=26)

    gt_mat_dilation = scipy.ndimage.binary_dilation(
        gt_mat, structure=dilation_struct, iterations=dil_factor)
    gt_mat_dilation_cc = cc3d.connected_components(
        gt_mat_dilation, connectivity=26)

    gt_mat_combinedByDilation = get_GTseg_combinedByDilation(
        gt_dilated_cc_mat=gt_mat_dilation_cc,
        gt_label_cc=gt_mat_cc
    )

    # Performing the Lesion-By-Lesion Comparison

    gt_label_cc = gt_mat_combinedByDilation
    pred_label_cc = pred_mat_cc

    gt_tp = []
    tp = []
    fn = []
    fp = []
    metric_pairs = []

    for gtcomp in range(np.max(gt_label_cc)):
        gtcomp += 1

        # Extracting current lesion
        gt_tmp = np.zeros_like(gt_label_cc)
        gt_tmp[gt_label_cc == gtcomp] = 1

        # Extracting ROI GT lesion component
        gt_tmp_dilation = scipy.ndimage.binary_dilation(
            gt_tmp, structure=dilation_struct, iterations=dil_factor)

        # Volume of lesion
        gt_vol = np.sum(gt_tmp)*sx*sy*sz

        # Extracting Predicted true positive lesions
        pred_tmp = np.copy(pred_label_cc)
        # pred_tmp = pred_tmp*gt_tmp
        pred_tmp = pred_tmp*gt_tmp_dilation
        intersecting_cc = np.unique(pred_tmp)
        intersecting_cc = intersecting_cc[intersecting_cc != 0]
        for cc in intersecting_cc:
            tp.append(cc)

        # Isolating Predited Lesions to calulcate Metrics
        pred_tmp = np.copy(pred_label_cc)
        pred_tmp[np.isin(pred_tmp, intersecting_cc, invert=True)] = 0
        pred_tmp[np.isin(pred_tmp, intersecting_cc)] = 1

        # Calculating Lesion-wise Dice and HD95
        dice_score = dice(pred_tmp, gt_tmp)
        surface_distances = surface_distance.compute_surface_distances(
            gt_tmp, pred_tmp, (sx, sy, sz))
        hd = surface_distance.compute_robust_hausdorff(surface_distances, 95)

        if np.any(gt_tmp > 0) and np.all(pred_tmp == 0):
            nsd_05 = 0
            nsd_10 = 0
        elif np.any(pred_tmp > 0) and np.all(gt_tmp == 0):
            nsd_05 = 0
            nsd_10 = 0
        elif np.all(pred_tmp == 0) and np.all(gt_tmp == 0):
            nsd_05 = 1
            nsd_10 = 1
        else:
            nsd_05 = normalized_surface_dice(
                pred_tmp,
                gt_tmp,
                threshold=0.5,
                spacing=(sx, sy, sz),
                connectivity=1
            )
            nsd_10 = normalized_surface_dice(
                pred_tmp,
                gt_tmp,
                threshold=1.0,
                spacing=(sx, sy, sz),
                connectivity=1
            )

        metric_pairs.append((intersecting_cc,
                            gtcomp, gt_vol, dice_score, hd, nsd_05, nsd_10))

        # Extracting Number of TP/FP/FN and other data
        if len(intersecting_cc) > 0:
            gt_tp.append(gtcomp)
        else:
            fn.append(gtcomp)

    fp = np.unique(
        pred_label_cc[np.isin(
            pred_label_cc, tp+[0], invert=True)])

    return tp, fn, fp, gt_tp, metric_pairs, full_dice, full_hd95, full_nsd_05, full_nsd_10, full_gt_vol, full_pred_vol, full_sens, full_specs


def get_sensitivity_and_specificity(result_array, target_array):
    """
    This function is extracted from GaNDLF from mlcommons

    You can find the documentation here - 

    https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/metrics/segmentation.py#L196

    """
    iC = np.sum(result_array)
    rC = np.sum(target_array)

    overlap = np.where((result_array == target_array), 1, 0)

    # Where they agree are both equal to that value
    TP = overlap[result_array == 1].sum()
    FP = iC - TP
    FN = rC - TP
    TN = np.count_nonzero((result_array != 1) & (target_array != 1))

    Sens = 1.0 * TP / (TP + FN + sys.float_info.min)
    Spec = 1.0 * TN / (TN + FP + sys.float_info.min)

    # Make Changes if both input and reference are 0 for the tissue type
    if (iC == 0) and (rC == 0):
        Sens = 1.0

    return Sens, Spec


def get_LesionWiseResults(pred_file, gt_file, challenge_name, output=None):
    """
    Computes the Lesion-wise scores for pair of prediction and ground truth
    segmentations

    Parameters
    ==========
    pred_file: str; location of the prediction segmentation    
    gt_file: str; location of the gt segmentation
    challenge_name: str; name of the challenge for parameters


    Output
    ======
    Saves the performance metrics as CSVs
    results_df: pd.DataFrame; lesion-wise results with other metrics
    """

    # Dilation and Threshold Parameters
    if challenge_name == 'BraTS-MEN':
        dilation_factor = 1
        lesion_volume_thresh = 50

    final_lesionwise_metrics_df = pd.DataFrame()
    final_metrics_dict = dict()
    label_values = ['WT', 'TC', 'ET']

    for l in range(len(label_values)):
        tp, fn, fp, gt_tp, metric_pairs, full_dice, full_hd95, full_nsd_05, full_nsd_10, full_gt_vol, full_pred_vol, full_sens, full_specs = get_LesionWiseScores(
            prediction_seg=pred_file,
            gt_seg=gt_file,
            label_value=label_values[l],
            dil_factor=dilation_factor
        )

        metric_df = pd.DataFrame(
            metric_pairs, columns=['predicted_lesion_numbers', 'gt_lesion_numbers',
                                   'gt_lesion_vol', 'dice_lesionwise', 'hd95_lesionwise', 'nsd05_lesionwise', 'nsd10_lesionwise']
        ).sort_values(by=['gt_lesion_numbers'], ascending=True).reset_index(drop=True)

        metric_df['_len'] = metric_df['predicted_lesion_numbers'].map(len)

        # Removing <= 50 lesions from analysis
        fn_sub = (metric_df[(metric_df['_len'] == 0) &
                  (metric_df['gt_lesion_vol'] <= lesion_volume_thresh)
        ]).shape[0]

        gt_tp_sub = (metric_df[(metric_df['_len'] != 0) &
                               (metric_df['gt_lesion_vol']
                                <= lesion_volume_thresh)
                               ]).shape[0]

        metric_df['Label'] = [label_values[l]]*len(metric_df)

        metric_df['hd95_lesionwise'] = metric_df['hd95_lesionwise'].replace(
            np.inf, 374)

        final_lesionwise_metrics_df = pd.concat(
            [final_lesionwise_metrics_df, metric_df],
            ignore_index=True
        )
        metric_df_thresh = metric_df[metric_df['gt_lesion_vol']
                                     > lesion_volume_thresh]

        try:
            lesion_wise_dice = np.sum(
                metric_df_thresh['dice_lesionwise'])/(len(metric_df_thresh) + len(fp))
        except:
            lesion_wise_dice = np.nan

        try:
            lesion_wise_nsd_05 = np.sum(
                metric_df_thresh['nsd05_lesionwise'])/(len(metric_df_thresh) + len(fp))
        except:
            lesion_wise_nsd_05 = np.nan

        try:
            lesion_wise_nsd_10 = np.sum(
                metric_df_thresh['nsd10_lesionwise'])/(len(metric_df_thresh) + len(fp))
        except:
            lesion_wise_nsd_10 = np.nan

        try:
            lesion_wise_hd95 = (np.sum(
                metric_df_thresh['hd95_lesionwise']) + len(fp)*374)/(len(metric_df_thresh) + len(fp))
        except:
            lesion_wise_hd95 = np.nan

        if math.isnan(lesion_wise_dice):
            lesion_wise_dice = 1

        if math.isnan(lesion_wise_nsd_05):
            lesion_wise_nsd_05 = 1

        if math.isnan(lesion_wise_nsd_10):
            lesion_wise_nsd_10 = 1

        if math.isnan(lesion_wise_hd95):
            lesion_wise_hd95 = 0

        metrics_dict = {
            'Num_TP': len(gt_tp) - gt_tp_sub,  # GT_TP
            # 'Num_TP' : len(tp),
            'Num_FP': len(fp),
            'Num_FN': len(fn) - fn_sub,
            'Sensitivity': full_sens,
            'Specificity': full_specs,
            'Legacy_Dice': full_dice,
            'Legacy_HD95': full_hd95,
            'Legacy NSD @ 0.5': full_nsd_05,
            'Legacy NSD @ 1.0': full_nsd_10,
            'GT_Complete_Volume': full_gt_vol,
            'LesionWise_Score_Dice': lesion_wise_dice,
            'LesionWise_Score_HD95': lesion_wise_hd95,
            'LesionWise_Score_NSD @ 0.5': lesion_wise_nsd_05,
            'LesionWise_Score_NSD @ 1.0': lesion_wise_nsd_10,
        }

        final_metrics_dict[label_values[l]] = metrics_dict

    # final_lesionwise_metrics_df.to_csv(os.path.split(pred_file)[0] + '/' +
    #                                   os.path.split(pred_file)[1].split('.')[0] +
    #                                   '_lesionwise_metrics.csv',
    #                                   index=False)

    results_df = pd.DataFrame(final_metrics_dict).T
    results_df['Labels'] = results_df.index
    results_df = results_df.reset_index(drop=True)
    results_df.insert(0, 'Labels', results_df.pop('Labels'))
    results_df.replace(np.inf, 374, inplace=True)

    if output:
        results_df.to_csv(output, index=False)

    return results_df, final_lesionwise_metrics_df
