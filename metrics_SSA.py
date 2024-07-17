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

    if tissue_type == 'WT':
        prediction_matrix = ((prediction_matrix >= 1) & (
            prediction_matrix <= 3)).astype(float)
        gt_matrix = ((gt_matrix >= 1) & (gt_matrix <= 3)).astype(float)

    elif tissue_type == 'TC':
        prediction_matrix = ((prediction_matrix == 1) | (
            prediction_matrix == 3)).astype(float)
        gt_matrix = ((gt_matrix == 1) | (gt_matrix == 3)).astype(float)

    elif tissue_type == 'NETC':
        prediction_matrix = (prediction_matrix == 1).astype(float)
        gt_matrix = (gt_matrix == 1).astype(float)

    elif tissue_type == 'ET':
        prediction_matrix = (prediction_matrix == 3).astype(float)
        gt_matrix = (gt_matrix == 3).astype(float)

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

    tissue_list = ["WT", "TC", "NETC", "ET"]
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

            if (t == "WT") | (t == "TC"):

                nib.save(
                    nib.Nifti1Image(gt_mat_combinedByDilation,
                                    affine=gt_affine),
                    f"./tmp_gt/{gt_file_name}/{gt_base}_{t}_cc_combined.nii.gz"
                )
            else:
                nib.save(
                    nib.Nifti1Image(gt_mat_combinedByDilation,
                                    affine=gt_affine),
                    f"./tmp_gt/{gt_file_name}/{gt_base}_{t}_cc.nii.gz"
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
            labels_to_remove = labels[counts <= lesion_volume_thresh]
            mask = np.isin(pred_mat_combinedByDilation,
                           labels_to_remove, invert=True)
            pred_mat_combinedByDilation = np.where(
                mask, pred_mat_combinedByDilation, 0)

            if (t == "WT") | (t == "TC"):

                nib.save(
                    nib.Nifti1Image(pred_mat_combinedByDilation,
                                    affine=pred_affine),
                    f"./tmp_pred/{pred_file_name}/{pred_base}_{t}_cc_combined.nii.gz"
                )
            else:
                nib.save(
                    nib.Nifti1Image(pred_mat_combinedByDilation,
                                    affine=pred_affine),
                    f"./tmp_pred/{pred_file_name}/{pred_base}_{t}_cc.nii.gz"
                )

        except Exception as e:
            print(f"Error processing {t}: {e}")


def get_touching_labels(nifti_img1, nifti_img2):
    img1 = nib.load(nifti_img1)
    img2 = nib.load(nifti_img2)

    data1 = img1.get_fdata()
    data2 = img2.get_fdata()

    structure = generate_binary_structure(3, 2)

    # Create binary masks for each label
    labels1 = np.unique(data1)[1:]  # Skip the background (label 0)
    labels2 = np.unique(data2)[1:]  # Skip the background (label 0)

    touching_labels = set()

    for label1 in labels1:
        mask1 = data1 == label1

        for label2 in labels2:
            mask2 = data2 == label2

            # Dilate the mask to check for touching
            dilated_mask1 = binary_dilation(mask1, structure)
            dilated_mask2 = binary_dilation(mask2, structure)

            if np.any(dilated_mask1 & mask2) or np.any(dilated_mask2 & mask1):
                touching_labels.add((label1, label2))

    '''
    
    print(labels1)

    if touching_labels:
        for label1, label2 in touching_labels:
            print(
                f"Label {label1} in image 1 touches label {label2} in image 2.")
    else:
        print("No labels from image 1 touch labels from image 2.")
    '''

    return touching_labels


def relabel_nifti_image(tuples, nifti_image_path, output_path):
    # Load the NIfTI image
    img = nib.load(nifti_image_path)
    img_data = img.get_fdata()

    # Determine the maximum label already in use in the image
    max_existing_label = int(np.max(img_data))

    # Create a mapping from original labels to new labels
    label_mapping = {}
    current_label = max_existing_label + 1  # Start assigning new labels from here
    seen_common_values = {}

    # First, map each common value to a new unique label not used in the image
    for original_label, common_value in sorted(tuples):
        if common_value not in seen_common_values:
            seen_common_values[common_value] = current_label
            current_label += 1

    # Then, map each original label to its new label based on the common value
    for original_label, common_value in sorted(tuples):
        if original_label not in label_mapping:
            label_mapping[original_label] = seen_common_values[common_value]

    # Relabel the image data
    relabeled_data = np.copy(img_data)
    for original_label, new_label in label_mapping.items():
        relabeled_data[img_data == original_label] = new_label

    # Save the relabeled image as a new NIfTI file
    new_img = nib.Nifti1Image(relabeled_data.astype(
        np.int32), img.affine, img.header)
    nib.save(new_img, output_path)
    # print(f'Relabeled image saved to {output_path}')

    # Print the label mapping for verification
    # print(label_mapping)


def reorder_labels_nifti(nifti_image_path, output_path):
    # Load the NIfTI image
    img = nib.load(nifti_image_path)
    img_data = img.get_fdata()

    # Find all unique labels in the image data
    unique_labels = np.unique(img_data)
    unique_labels.sort()  # Sort the labels to maintain consistent ordering

    # Remove the background label if it's zero and should not be reassigned
    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]

    # Set a high starting label to avoid conflicts with existing labels
    start_label = int(np.max(img_data)) + 1

    # First pass: assign temporary labels starting from a high number to avoid overlap
    temp_label_mapping = {label: i + start_label for i,
                          label in enumerate(unique_labels)}
    temp_relabel_data = np.copy(img_data)
    for original_label, temp_label in temp_label_mapping.items():
        temp_relabel_data[img_data == original_label] = temp_label

    # Second pass: map temporary labels to final labels starting from 1
    final_label_mapping = {temp_label: i + 1 for i,
                           temp_label in enumerate(sorted(temp_label_mapping.values()))}
    final_relabel_data = np.copy(temp_relabel_data)
    for temp_label, final_label in final_label_mapping.items():
        final_relabel_data[temp_relabel_data == temp_label] = final_label

    # Ensure the final label data is cast to an integer type to fix any floating point issues
    final_relabel_data = final_relabel_data.astype(np.int32)

    # Save the relabeled image as a new NIfTI file
    new_img = nib.Nifti1Image(final_relabel_data, img.affine, img.header)
    nib.save(new_img, output_path)

    # print(f'Reordered label image saved to {output_path}')

    # Print the label mappings for verification
    # print("Temporary Label Mapping:", temp_label_mapping)
    # print("Final Label Mapping:", final_label_mapping)


def combine_lesions_ET(et_cc, netc_cc):
    op_path = os.path.join(
        os.path.split(et_cc)[0],
        os.path.split(et_cc)[1].split(".")[0] + "_combined.nii.gz"
    )

    touching_labels = get_touching_labels(et_cc, netc_cc)
    if len(touching_labels) > 1:
        relabel_nifti_image(touching_labels,
                            nifti_image_path=et_cc,
                            output_path=op_path)

    else:
        shutil.copy(et_cc, op_path)


def combine_lesions_NETC(netc_cc):
    op_path = os.path.join(
        os.path.split(netc_cc)[0],
        os.path.split(netc_cc)[1].split(".")[0] + "_combined.nii.gz"
    )
    shutil.copy(netc_cc, op_path)


def combine_lesions_tissues(netc_cc, et_cc):

    combine_lesions_ET(et_cc, netc_cc)
    combine_lesions_NETC(netc_cc)

    # print("Combining lesions complete!")


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
    if challenge_name == 'BraTS-SSA':
        dilation_factor = 3
        lesion_volume_thresh = 50

    final_lesionwise_metrics_df = pd.DataFrame()
    final_metrics_dict = dict()
    label_values = ['WT', 'TC', 'ET']

    save_tmp_files(pred_file=pred_file, gt_file=gt_file,
                   dil_factor=dilation_factor)

    gt_file_name = gt_file.split('/')[-1].split('.')[0]
    et_cc = f"./tmp_gt/{gt_file_name}/{gt_file_name}_ET_cc.nii.gz"
    netc_cc = f"./tmp_gt/{gt_file_name}/{gt_file_name}_NETC_cc.nii.gz"

    combine_lesions_tissues(netc_cc, et_cc)

    reorder_labels_nifti(
        nifti_image_path=os.path.join(
            os.path.split(et_cc)[0],
            os.path.split(et_cc)[1].split(".")[0] + "_combined.nii.gz"
        ),
        output_path=os.path.join(
            os.path.split(et_cc)[0],
            os.path.split(et_cc)[1].split(".")[0] + "_combined.nii.gz"
        )
    )

    pred_file_name = pred_file.split('/')[-1].split('.')[0]
    et_cc = f"./tmp_pred/{pred_file_name}/{pred_file_name}_ET_cc.nii.gz"
    netc_cc = f"./tmp_pred/{pred_file_name}/{pred_file_name}_NETC_cc.nii.gz"
    combine_lesions_tissues(netc_cc, et_cc)

    reorder_labels_nifti(
        nifti_image_path=os.path.join(
            os.path.split(et_cc)[0],
            os.path.split(et_cc)[1].split(".")[0] + "_combined.nii.gz"
        ),
        output_path=os.path.join(
            os.path.split(et_cc)[0],
            os.path.split(et_cc)[1].split(".")[0] + "_combined.nii.gz"
        )
    )

    for l in range(len(label_values)):

        tp, fn, fp, gt_tp, metric_pairs, full_dice, full_hd95, full_gt_vol, full_pred_vol, full_sens, full_specs = get_LesionWiseScores(
            prediction_seg=pred_file,
            gt_seg=gt_file,
            label_value=label_values[l],
            dil_factor=dilation_factor
        )

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
        metric_df = metric_df.replace(np.inf, 374)

        final_lesionwise_metrics_df = final_lesionwise_metrics_df.append(
            metric_df)
        metric_df_thresh = metric_df[metric_df['gt_lesion_vol']
                                     > lesion_volume_thresh]

        try:
            lesion_wise_dice = np.sum(
                metric_df_thresh['dice_lesionwise'])/(len(metric_df_thresh) + len(fp))
        except:
            lesion_wise_dice = np.nan

        try:
            lesion_wise_hd95 = (np.sum(
                metric_df_thresh['hd95_lesionwise']) + len(fp)*374)/(len(metric_df_thresh) + len(fp))
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
    results_df.replace(np.inf, 374, inplace=True)

    if output:
        results_df.to_csv(output, index=False)

    return results_df, final_lesionwise_metrics_df
