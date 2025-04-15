import numpy as np
import sys

_SSA_LABEL_RULES = {
    'WT': lambda x: (x >= 1) & (x <= 3),
    'TC': lambda x: (x == 1) | (x == 3),
    'NETC': lambda x: (x == 1),
    'ET': lambda x: (x == 3)
}

_PED_LABEL_RULES = {
    'WT': lambda x: (x >= 1) & (x <= 4),
    'TC': lambda x: (x >= 1) & (x <= 3),
    'ET': lambda x: (x == 1),
    'NETC': lambda x: (x == 2),
    'CC': lambda x: (x == 3),
    'ED': lambda x: (x == 4)
}


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


def get_label_rules(dataset_name):
    """
    Returns label rules for a given dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'SSA', 'PED').

    Returns
    -------
    dict
        Mapping of tissue types to label masks.
    """
    dataset_name = dataset_name.upper()

    if dataset_name == 'SSA':
        return _SSA_LABEL_RULES
    elif dataset_name == 'PED':
        return _PED_LABEL_RULES
    else:
        raise ValueError(f"Unknown dataset name '{dataset_name}'")


def get_TissueWiseSeg(pred_matrix, gt_matrix, tissue_type, label_rules):
    """
    General-purpose tissue-wise segmentation extractor.

    Parameters
    ----------
    pred_matrix : np.ndarray
        Predicted segmentation array.
    gt_matrix : np.ndarray
        Ground truth segmentation array.
    tissue_type : str
        Name of the tissue type to extract.
    label_rules : dict
        Mapping from tissue_type string to a lambda or function that takes a matrix and returns a binary mask.

    Returns
    -------
    pred_out : np.ndarray
        Binary mask for predicted matrix.
    gt_out : np.ndarray
        Binary mask for ground truth matrix.
    """
    if tissue_type not in label_rules:
        raise ValueError(
            f"Tissue type '{tissue_type}' not defined in label_rules.")

    pred_out = label_rules[tissue_type](pred_matrix)
    gt_out = label_rules[tissue_type](gt_matrix)

    return pred_out.astype(float), gt_out.astype(float)
