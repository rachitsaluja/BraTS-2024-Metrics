import numpy as np
import nibabel as nib
import scipy
import os
import pandas as pd
import surface_distance
import math
import shutil
from BraTS_SegMetrics.utils import dice, get_sensitivity_and_specificity
from BraTS_SegMetrics.utils import get_TissueWiseSeg, get_label_rules
from BraTS_SegMetrics.processing import (
    get_touching_labels, process_tissue_component, relabel_nifti_image, reorder_labels_nifti
)


def save_tmp_files(pred_file, gt_file, dil_factor):
    tissue_list = ["WT", "TC", "NETC", "ET"]
    label_rules = get_label_rules("SSA")

    pred_nii = nib.load(pred_file)
    gt_nii = nib.load(gt_file)
    pred_mat = pred_nii.get_fdata()
    gt_mat = gt_nii.get_fdata()

    pred_affine = pred_nii.affine
    gt_affine = gt_nii.affine

    pred_base = os.path.splitext(
        os.path.splitext(os.path.basename(pred_file))[0])[0]
    gt_base = os.path.splitext(
        os.path.splitext(os.path.basename(gt_file))[0])[0]

    pred_out_dir = f"./tmp_pred/{pred_base}"
    gt_out_dir = f"./tmp_gt/{gt_base}"
    os.makedirs(pred_out_dir, exist_ok=True)
    os.makedirs(gt_out_dir, exist_ok=True)

    for t in tissue_list:
        try:
            # Tissue-wise segmentation
            pred_tissue, gt_tissue = get_TissueWiseSeg(
                pred_mat, gt_mat, t, label_rules)

            pred_path = os.path.join(pred_out_dir, f"{pred_base}_{t}.nii.gz")
            gt_path = os.path.join(gt_out_dir, f"{gt_base}_{t}.nii.gz")

            nib.save(nib.Nifti1Image(pred_tissue, pred_affine), pred_path)
            nib.save(nib.Nifti1Image(gt_tissue, gt_affine), gt_path)

        except Exception as e:
            print(f"[Seg Error] {t}: {e}")

    for t in tissue_list:
        try:
            # Ground truth processing
            gt_img = nib.load(os.path.join(
                gt_out_dir, f"{gt_base}_{t}.nii.gz"))
            gt_out_img = process_tissue_component(
                gt_img.get_fdata(), gt_img.affine, dil_factor)

            suffix = "_cc_combined.nii.gz" if t in (
                "WT", "TC") else "_cc.nii.gz"
            nib.save(gt_out_img, os.path.join(
                gt_out_dir, f"{gt_base}_{t}{suffix}"))

        except Exception as e:
            print(f"[GT Error] {t}: {e}")

    for t in tissue_list:
        try:
            # Prediction processing
            pred_img = nib.load(os.path.join(
                pred_out_dir, f"{pred_base}_{t}.nii.gz"))
            pred_out_img = process_tissue_component(
                pred_img.get_fdata(), pred_img.affine, dil_factor, lesion_volume_thresh=50)

            suffix = "_cc_combined.nii.gz" if t in (
                "WT", "TC") else "_cc.nii.gz"
            nib.save(pred_out_img, os.path.join(
                pred_out_dir, f"{pred_base}_{t}{suffix}"))

        except Exception as e:
            print(f"[Pred Error] {t}: {e}")


def get_combined_output_path(nifti_path):
    """
    Creates an output path for the combined image by appending '_combined' to the filename.
    """
    base_dir = os.path.dirname(nifti_path)
    base_name = os.path.basename(nifti_path).split(".")[0]
    return os.path.join(base_dir, f"{base_name}_combined.nii.gz")


def combine_lesions_ET(et_cc, netc_cc):
    """
    Combines ET lesions by relabeling overlapping lesions with NETC.
    If no touching lesions found, simply copies ET file as output.
    """
    output_path = get_combined_output_path(et_cc)

    try:
        touching_labels = get_touching_labels(et_cc, netc_cc)

        if touching_labels:
            relabel_nifti_image(
                touching_labels, nifti_image_path=et_cc, output_path=output_path)
        else:
            shutil.copy(et_cc, output_path)

    except Exception as e:
        print(f"[Error] combine_lesions_ET failed: {e}")
        shutil.copy(et_cc, output_path)  # Fallback: just copy input

    return output_path


def combine_lesions_NETC(netc_cc):
    """
    Simply copies the NETC segmentation as a 'combined' output.
    """
    output_path = get_combined_output_path(netc_cc)
    try:
        shutil.copy(netc_cc, output_path)
    except Exception as e:
        print(f"[Error] combine_lesions_NETC failed: {e}")

    return output_path


def combine_lesions_tissues(netc_cc, et_cc):
    """
    Wrapper function to combine ET and NETC lesions.
    """
    et_combined = combine_lesions_ET(et_cc, netc_cc)
    netc_combined = combine_lesions_NETC(netc_cc)
    return et_combined, netc_combined


def get_LesionWiseScores(prediction_seg, gt_seg, label_value, dil_factor):
    """
    Computes lesion-wise and full-image segmentation scores (Dice, HD95, Sensitivity, Specificity)
    for a given label type using dilated matching between ground truth and prediction.

    Parameters
    ----------
    prediction_seg : str
        Path to the predicted segmentation (NIfTI).
    gt_seg : str
        Path to the ground truth segmentation (NIfTI).
    label_value : str
        Tissue label string (e.g., 'WT', 'TC').
    dil_factor : int
        Dilation factor (iterations) for lesion matching.

    Returns
    -------
    tp : list
        List of predicted CC labels considered true positive.
    fn : list
        List of missed ground truth lesion component IDs.
    fp : list
        List of predicted components not overlapping with any GT lesion.
    gt_tp : list
        Ground truth lesion IDs that were successfully matched.
    metric_pairs : list of tuples
        (predicted_ccs, gt_lesion_id, gt_volume, dice, hd95) for each matched lesion.
    full_dice : float
        Dice score for entire volume.
    full_hd95 : float
        HD95 score for entire volume.
    full_gt_vol : float
        Volume of GT lesions.
    full_pred_vol : float
        Volume of predicted lesions.
    full_sens : float
        Sensitivity of full prediction.
    full_specs : float
        Specificity of full prediction.
    """
    pred_base = os.path.basename(prediction_seg).split('.')[0]
    gt_base = os.path.basename(gt_seg).split('.')[0]

    pred_path = f"./tmp_pred/{pred_base}/{pred_base}_{label_value}_cc_combined.nii.gz"
    gt_path = f"./tmp_gt/{gt_base}/{gt_base}_{label_value}_cc_combined.nii.gz"

    pred_nii = nib.load(pred_path)
    gt_nii = nib.load(gt_path)

    pred_mat = pred_nii.get_fdata()
    gt_mat = gt_nii.get_fdata()

    sx, sy, sz = pred_nii.header.get_zooms()

    # Volume (mm³) of GT and prediction
    full_gt_vol = np.sum(gt_mat > 0) * sx * sy * sz
    full_pred_vol = np.sum(pred_mat > 0) * sx * sy * sz

    # Full-image Dice
    if np.all(gt_mat == 0) and np.all(pred_mat == 0):
        full_dice = 1.0
        full_hd95 = 0.0
    else:
        full_dice = dice(pred_mat, gt_mat)
        surface_dist = surface_distance.compute_surface_distances(gt_mat.astype(int),
                                                                  pred_mat.astype(
                                                                      int),
                                                                  (sx, sy, sz))
        full_hd95 = surface_distance.compute_robust_hausdorff(surface_dist, 95)

    full_sens, full_specs = get_sensitivity_and_specificity(pred_mat, gt_mat)

    # Connected components
    gt_label_cc = gt_mat.astype(np.int32)
    pred_label_cc = pred_mat.astype(np.int32)
    dilation_struct = scipy.ndimage.generate_binary_structure(3, 2)

    tp, fn, fp, gt_tp, metric_pairs = [], [], [], [], []

    for gtcomp in range(1, np.max(gt_label_cc) + 1):
        gt_tmp = (gt_label_cc == gtcomp).astype(np.uint8)
        gt_tmp_dilated = scipy.ndimage.binary_dilation(
            gt_tmp, structure=dilation_struct, iterations=dil_factor)

        # Match predicted components overlapping with dilated GT
        intersecting_cc = np.unique(pred_label_cc[gt_tmp_dilated])
        # remove background
        intersecting_cc = intersecting_cc[intersecting_cc != 0]

        # Volume
        gt_vol = np.sum(gt_tmp) * sx * sy * sz

        if intersecting_cc.size > 0:
            gt_tp.append(gtcomp)
            tp.extend(intersecting_cc)

            # Binary prediction mask
            pred_mask = np.isin(
                pred_label_cc, intersecting_cc).astype(np.uint8)

            dice_score = dice(pred_mask, gt_tmp)
            hd95 = surface_distance.compute_robust_hausdorff(
                surface_distance.compute_surface_distances(gt_tmp, pred_mask, (sx, sy, sz)), 95)

            metric_pairs.append(
                (intersecting_cc.tolist(), gtcomp, gt_vol, dice_score, hd95))
        else:
            fn.append(gtcomp)

    # Identify FP lesions (predictions that didn't match any GT lesion)
    tp = list(np.unique(tp))
    fp = list(np.unique(pred_label_cc[np.isin(
        pred_label_cc, tp + [0], invert=True)]))

    return tp, fn, fp, gt_tp, metric_pairs, full_dice, full_hd95, full_gt_vol, full_pred_vol, full_sens, full_specs


def get_LesionWiseResults(pred_file: str, gt_file: str, challenge_name: str, output: str = None):
    """
    Computes the lesion-wise and full-volume scores for a pair of prediction and
    ground truth segmentations, saving CSV outputs if specified.

    Parameters
    ----------
    pred_file : str
        Path to the predicted segmentation file.
    gt_file : str
        Path to the ground truth segmentation file.
    challenge_name : str
        Challenge name to determine evaluation parameters (e.g., "BraTS-SSA").
    output : str, optional
        Path to save summary results CSV.

    Returns
    -------
    results_df : pd.DataFrame
        Summary metrics per label.
    final_lesionwise_metrics_df : pd.DataFrame
        Lesion-wise metric table.
    """

    if challenge_name == 'BraTS-SSA':
        dilation_factor = 3
        lesion_volume_thresh = 50
    else:
        raise ValueError(f"Unsupported challenge: {challenge_name}")

    label_values = ['WT', 'TC', 'ET']
    final_lesionwise_metrics_df = pd.DataFrame()
    final_metrics_dict = {}

    save_tmp_files(pred_file=pred_file, gt_file=gt_file,
                   dil_factor=dilation_factor)

    for mode, file in zip(["gt", "pred"], [gt_file, pred_file]):
        base = os.path.basename(file).split('.')[0]
        et_cc = f"./tmp_{mode}/{base}/{base}_ET_cc.nii.gz"
        netc_cc = f"./tmp_{mode}/{base}/{base}_NETC_cc.nii.gz"

        combine_lesions_tissues(netc_cc, et_cc)
        reorder_labels_nifti(
            nifti_image_path=get_combined_output_path(et_cc),
            output_path=get_combined_output_path(et_cc)
        )

    for label in label_values:
        (
            tp, fn, fp, gt_tp,
            metric_pairs,
            full_dice, full_hd95,
            full_gt_vol, full_pred_vol,
            full_sens, full_specs
        ) = get_LesionWiseScores(
            prediction_seg=pred_file,
            gt_seg=gt_file,
            label_value=label,
            dil_factor=dilation_factor
        )

        metric_df = pd.DataFrame(
            metric_pairs,
            columns=['predicted_lesion_numbers', 'gt_lesion_numbers',
                     'gt_lesion_vol', 'dice_lesionwise', 'hd95_lesionwise']
        ).sort_values(by='gt_lesion_numbers').reset_index(drop=True)

        metric_df['_len'] = metric_df['predicted_lesion_numbers'].map(len)

        fn_sub = ((metric_df['_len'] == 0) & (
            metric_df['gt_lesion_vol'] <= lesion_volume_thresh)).sum()
        gt_tp_sub = ((metric_df['_len'] != 0) & (
            metric_df['gt_lesion_vol'] <= lesion_volume_thresh)).sum()

        metric_df['Label'] = label
        metric_df = metric_df.replace(np.inf, 374)

        final_lesionwise_metrics_df = pd.concat(
            [final_lesionwise_metrics_df, metric_df], ignore_index=True)
        metric_df_thresh = metric_df[metric_df['gt_lesion_vol']
                                     > lesion_volume_thresh]

        lesion_denom = len(metric_df_thresh) + len(fp)

        lesion_wise_dice = np.nan
        lesion_wise_hd95 = np.nan

        if lesion_denom > 0:
            if len(metric_df_thresh) > 0:
                lesion_wise_dice = np.sum(
                    metric_df_thresh['dice_lesionwise']) / lesion_denom
                lesion_wise_hd95 = (
                    np.sum(metric_df_thresh['hd95_lesionwise']) + len(fp) * 374
                ) / lesion_denom
            else:
                lesion_wise_dice = 0
                lesion_wise_hd95 = 374

        if math.isnan(lesion_wise_dice):
            lesion_wise_dice = 1.0
        if math.isnan(lesion_wise_hd95):
            lesion_wise_hd95 = 0.0

        metrics_dict = {
            'Num_TP': len(gt_tp) - gt_tp_sub,
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

        final_metrics_dict[label] = metrics_dict

    results_df = pd.DataFrame(final_metrics_dict).T.reset_index().rename(
        columns={"index": "Labels"})
    results_df.replace(np.inf, 374, inplace=True)

    if output:
        results_df.to_csv(output, index=False)

    return results_df, final_lesionwise_metrics_df
