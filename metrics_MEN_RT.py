import numpy as np
import nibabel as nib
import cc3d
import scipy
import os
import pandas as pd
import surface_distance
import sys
import math
from scipy.ndimage import label, generate_binary_structure, binary_dilation
import shutil


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

    if tissue_type == 'GTV':
        prediction_matrix = (prediction_matrix == 1).astype(float)
        gt_matrix = (gt_matrix == 1).astype(float)

    return prediction_matrix, gt_matrix


def get_Predseg_combinedByDilation(pred_dilated_cc_mat, pred_label_cc):
    """
    Computes the Corrected Connected Components after combing lesions
    together with respect to their dilation extent

    Parameters
    ==========
    pred_dilated_cc_mat: Numpy Array/Matrix; Ground Truth Dilated Segmentation 
                       after CC Analysis
    pred_label_cc: Numpy Array/Matrix; Ground Truth Segmentation after 
                       CC Analysis

    Output
    ======
    pred_seg_combinedByDilation_mat: Numpy Array/Matrix; Ground Truth 
                                   Segmentation after CC Analysis and 
                                   combining lesions
    """

    pred_seg_combinedByDilation_mat = np.zeros_like(pred_dilated_cc_mat)

    for comp in range(np.max(pred_dilated_cc_mat)):
        comp += 1

        pred_d_tmp = np.zeros_like(pred_dilated_cc_mat)
        pred_d_tmp[pred_dilated_cc_mat == comp] = 1
        pred_d_tmp = (pred_label_cc*pred_d_tmp)

        np.place(pred_d_tmp, pred_d_tmp > 0, comp)
        pred_seg_combinedByDilation_mat += pred_d_tmp

    return pred_seg_combinedByDilation_mat


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


def save_tmp_files(pred_file, gt_file, dil_factor):

    gt_file_name = gt_file.split('/')[-1].split('.')[0]
    pred_file_name = pred_file.split('/')[-1].split('.')[0]

    os.makedirs(f"./tmp_gt/{gt_file_name}", exist_ok=True)
    os.makedirs(f"./tmp_pred/{pred_file_name}", exist_ok=True)

    pred_nii = nib.load(pred_file)
    gt_nii = nib.load(gt_file)

    pred_mat = pred_nii.get_fdata()
    gt_mat = gt_nii.get_fdata()

    pred_base = os.path.splitext(
        os.path.splitext(os.path.basename(pred_file))[0])[0]
    gt_base = os.path.splitext(
        os.path.splitext(os.path.basename(gt_file))[0])[0]

    tissue_list = ["GTV"]
    for t in tissue_list:
        try:
            pred_tissue_mat, gt_tissue_mat = get_TissueWiseSeg(prediction_matrix=pred_mat,
                                                               gt_matrix=gt_mat,
                                                               tissue_type=t)
            nib.save(
                nib.Nifti1Image(pred_tissue_mat, affine=pred_nii.affine),
                f"./tmp_pred/{pred_file_name}/{pred_base}_{t}.nii.gz"
            )
            nib.save(
                nib.Nifti1Image(gt_tissue_mat, affine=gt_nii.affine),
                f"./tmp_gt/{gt_file_name}/{gt_base}_{t}.nii.gz"
            )
        except Exception as e:
            print(f"Error processing {t}: {e}")

    for t in tissue_list:
        try:
            gt_mat = nib.load(
                f"./tmp_gt/{gt_file_name}/{gt_base}_{t}.nii.gz").get_fdata()
            gt_affine = nib.load(
                f"./tmp_gt/{gt_file_name}/{gt_base}_{t}.nii.gz").affine
            dilation_struct = scipy.ndimage.generate_binary_structure(3, 2)

            gt_mat_cc = cc3d.connected_components(gt_mat, connectivity=26)
            gt_mat_dilation = scipy.ndimage.binary_dilation(
                gt_mat, structure=dilation_struct, iterations=dil_factor)
            gt_mat_dilation_cc = cc3d.connected_components(
                gt_mat_dilation, connectivity=26)

            gt_mat_combinedByDilation = get_GTseg_combinedByDilation(
                gt_dilated_cc_mat=gt_mat_dilation_cc,
                gt_label_cc=gt_mat_cc)

            if (t == "GTV"):

                nib.save(
                    nib.Nifti1Image(gt_mat_combinedByDilation,
                                    affine=gt_affine),
                    f"./tmp_gt/{gt_file_name}/{gt_base}_{t}_cc_combined.nii.gz"
                )

        except Exception as e:
            print(f"Error processing {t}: {e}")

    for t in tissue_list:
        try:
            lesion_volume_thresh = 50
            pred_mat = nib.load(
                f"./tmp_pred/{pred_file_name}/{pred_base}_{t}.nii.gz").get_fdata()
            pred_affine = nib.load(
                f"./tmp_pred/{pred_file_name}/{pred_base}_{t}.nii.gz").affine
            dilation_struct = scipy.ndimage.generate_binary_structure(3, 2)
            sx, sy, sz = nib.load(
                f"./tmp_pred/{pred_file_name}/{pred_base}_{t}.nii.gz").header.get_zooms()

            pred_mat_cc = cc3d.connected_components(pred_mat, connectivity=26)
            pred_mat_dilation = scipy.ndimage.binary_dilation(
                pred_mat, structure=dilation_struct, iterations=dil_factor)
            pred_mat_dilation_cc = cc3d.connected_components(
                pred_mat_dilation, connectivity=26)

            pred_mat_combinedByDilation = get_Predseg_combinedByDilation(
                pred_dilated_cc_mat=pred_mat_dilation_cc,
                pred_label_cc=pred_mat_cc)

            labels, counts = np.unique(
                pred_mat_combinedByDilation, return_counts=True)
            labels_to_remove = labels[counts <= (
                lesion_volume_thresh)/(sx*sy*sz)]
            mask = np.isin(pred_mat_combinedByDilation,
                           labels_to_remove, invert=True)
            pred_mat_combinedByDilation = np.where(
                mask, pred_mat_combinedByDilation, 0)

            if (t == "GTV"):

                nib.save(
                    nib.Nifti1Image(pred_mat_combinedByDilation,
                                    affine=pred_affine),
                    f"./tmp_pred/{pred_file_name}/{pred_base}_{t}_cc_combined.nii.gz"
                )

        except Exception as e:
            print(f"Error processing {t}: {e}")


def get_LesionWiseScores(prediction_seg, gt_seg, label_value, dil_factor):

    pred_file_name = prediction_seg.split('/')[-1].split('.')[0]
    pred_nii = nib.load(
        f"./tmp_pred/{pred_file_name}/{pred_file_name}_{label_value}_cc_combined.nii.gz")

    gt_file_name = gt_seg.split('/')[-1].split('.')[0]
    gt_nii = nib.load(
        f"./tmp_gt/{gt_file_name}/{gt_file_name}_{label_value}_cc_combined.nii.gz")

    pred_mat = pred_nii.get_fdata()
    gt_mat = gt_nii.get_fdata()

    sx, sy, sz = pred_nii.header.get_zooms()

    # Get Dice score for the full image
    if np.all(gt_mat == 0) and np.all(pred_mat == 0):
        full_dice = 1.0
    else:
        full_dice = dice(
            pred_mat,
            gt_mat
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

    dilation_struct = scipy.ndimage.generate_binary_structure(3, 2)

    # Get GT Volume and Pred Volume for the full image
    full_gt_vol = np.sum(gt_mat > 0)*sx*sy*sz
    full_pred_vol = np.sum(pred_mat > 0)*sx*sy*sz

    pred_mat_cc = pred_mat

    gt_label_cc = gt_mat.astype(
        np.int32)
    pred_label_cc = pred_mat_cc.astype(
        np.int32)

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
        gt_vol = np.sum(gt_tmp > 0)*sx*sy*sz

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

        metric_pairs.append((intersecting_cc,
                            gtcomp, gt_vol, dice_score, hd))

        # Extracting Number of TP/FP/FN and other data
        if len(intersecting_cc) > 0:
            gt_tp.append(gtcomp)
        else:
            fn.append(gtcomp)

    fp = np.unique(
        pred_label_cc[np.isin(
            pred_label_cc, tp+[0], invert=True)])

    return tp, fn, fp, gt_tp, metric_pairs, full_dice, full_hd95, full_gt_vol, full_pred_vol, full_sens, full_specs


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
    if challenge_name == 'BraTS-MEN-RT':
        dilation_factor = 1
        lesion_volume_thresh = 50

    final_lesionwise_metrics_df = pd.DataFrame()
    final_metrics_dict = dict()
    label_values = ['GTV']

    save_tmp_files(pred_file=pred_file, gt_file=gt_file,
                   dil_factor=dilation_factor)

    for l in range(len(label_values)):

        tp, fn, fp, gt_tp, metric_pairs, full_dice, full_hd95, full_gt_vol, full_pred_vol, full_sens, full_specs = get_LesionWiseScores(
            prediction_seg=pred_file,
            gt_seg=gt_file,
            label_value=label_values[l],
            dil_factor=dilation_factor
        )

        shape_img = nib.load(pred_file).get_fdata().shape
        hd_value = math.sqrt(
            (shape_img[0]**2) + (shape_img[1]**2) + (shape_img[2]**2))

        metric_df = pd.DataFrame(
            metric_pairs, columns=['predicted_lesion_numbers', 'gt_lesion_numbers',
                                   'gt_lesion_vol', 'dice_lesionwise', 'hd95_lesionwise']
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
        metric_df = metric_df.replace(np.inf, hd_value)

        final_lesionwise_metrics_df = final_lesionwise_metrics_df.append(
            metric_df)
        metric_df_thresh = metric_df[metric_df['gt_lesion_vol']
                                     > lesion_volume_thresh]

        try:
            # Removing FP for MEN-RT Challenge
            lesion_wise_dice = np.sum(
                metric_df_thresh['dice_lesionwise'])/(len(metric_df_thresh))
        except:
            lesion_wise_dice = np.nan

        try:
            # Removing FP for MEN-RT Challenge
            lesion_wise_hd95 = (np.sum(
                metric_df_thresh['hd95_lesionwise']))/(len(metric_df_thresh))
        except:
            lesion_wise_hd95 = np.nan

        if math.isnan(lesion_wise_dice):
            lesion_wise_dice = 1

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
            'GT_Complete_Volume': full_gt_vol,
            'LesionWise_Score_Dice': lesion_wise_dice,
            'LesionWise_Score_HD95': lesion_wise_hd95
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
    results_df.replace(np.inf, hd_value, inplace=True)

    if output:
        results_df.to_csv(output, index=False)

    return results_df, final_lesionwise_metrics_df
