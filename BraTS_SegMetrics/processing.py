import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, generate_binary_structure
import cc3d


def combine_cc_by_dilation(dilated_cc_mat, label_cc_mat):
    """
    Computes the corrected connected components after combining lesions
    based on their dilation extent.

    This function applies to both prediction and ground truth segmentation 
    maps by generalizing the input variable names.

    Parameters
    ==========
    dilated_cc_mat : np.ndarray
        Dilated segmentation map where each connected component has
        a unique integer label (after CC analysis and dilation).

    label_cc_mat : np.ndarray
        Original connected component segmentation map before dilation.

    Returns
    =======
    combined_segmentation : np.ndarray
        Segmentation map where components from label_cc_mat are reassigned 
        based on their overlap with each dilated component in dilated_cc_mat.
    """

    combined_segmentation = np.zeros_like(dilated_cc_mat)

    components = np.unique(dilated_cc_mat)
    components = components[components != 0]

    for comp in components:
        mask = dilated_cc_mat == comp

        overlapping_labels = np.unique(label_cc_mat[mask])
        overlapping_labels = overlapping_labels[overlapping_labels != 0]

        for label in overlapping_labels:
            combined_segmentation[label_cc_mat == label] = comp

    return combined_segmentation


def get_touching_labels(nifti_img1, nifti_img2):
    """
    Identifies label pairs from two NIfTI segmentation images that spatially touch.

    Parameters
    ----------
    nifti_img1 : str
        Path to the first NIfTI file.
    nifti_img2 : str
        Path to the second NIfTI file.

    Returns
    -------
    touching_labels : set of tuples
        Set of (label1, label2) pairs where label1 from image1 touches label2 from image2.
    """
    img1 = nib.load(nifti_img1)
    img2 = nib.load(nifti_img2)

    data1 = img1.get_fdata().astype(np.int32)
    data2 = img2.get_fdata().astype(np.int32)

    structure = generate_binary_structure(3, 2)

    labels1 = np.unique(data1)
    labels2 = np.unique(data2)
    labels1 = labels1[labels1 != 0]
    labels2 = labels2[labels2 != 0]

    touching_labels = set()

    img2_binary = data2 > 0

    for label1 in labels1:
        mask1 = data1 == label1
        dilated_mask1 = binary_dilation(mask1, structure)

        overlap = dilated_mask1 & img2_binary
        overlapping_labels = np.unique(data2[overlap])
        overlapping_labels = overlapping_labels[overlapping_labels != 0]

        for label2 in overlapping_labels:
            touching_labels.add((label1, label2))

    return touching_labels


def process_tissue_component(tissue_mat, affine, dil_factor, lesion_volume_thresh=None):
    """
    Common logic for CC + Dilation + Combining + Filtering small lesions (optional).
    """
    dilation_struct = generate_binary_structure(3, 2)

    cc = cc3d.connected_components(tissue_mat, connectivity=26)
    dilated = binary_dilation(
        tissue_mat, structure=dilation_struct, iterations=dil_factor)
    dilated_cc = cc3d.connected_components(dilated, connectivity=26)

    combined = combine_cc_by_dilation(dilated_cc, cc)

    if lesion_volume_thresh is not None:
        labels, counts = np.unique(combined, return_counts=True)
        labels_to_remove = labels[counts <= lesion_volume_thresh]
        combined = np.where(
            np.isin(combined, labels_to_remove, invert=True), combined, 0)

    return nib.Nifti1Image(combined, affine)


def relabel_nifti_image(tuples, nifti_image_path, output_path):
    """
    Relabels a NIfTI image based on a list of (original_label, common_value) pairs.
    Labels sharing the same common_value will be merged into a new unique label.

    Parameters
    ----------
    tuples : list of tuples
        Each tuple is (original_label, common_value). All original_labels with the same 
        common_value will get the same new label.

    nifti_image_path : str
        Path to the input NIfTI image.

    output_path : str
        Path to save the relabeled NIfTI image.

    Returns
    -------
    dict
        A dictionary mapping original labels to new labels.
    """
    img = nib.load(nifti_image_path)
    img_data = img.get_fdata()

    max_existing_label = int(np.max(img_data))
    current_label = max_existing_label + 1

    label_mapping = {}
    common_value_map = {}
    for original_label, common_value in sorted(tuples):
        if common_value not in common_value_map:
            common_value_map[common_value] = current_label
            current_label += 1
        label_mapping[original_label] = common_value_map[common_value]

    relabeled_data = np.copy(img_data)
    for original_label, new_label in label_mapping.items():
        if original_label != new_label:  # Optional optimization
            relabeled_data[img_data == original_label] = new_label

    relabeled_img = nib.Nifti1Image(
        relabeled_data.astype(np.int32), img.affine, img.header)
    nib.save(relabeled_img, output_path)

    return label_mapping


def reorder_labels_nifti(nifti_image_path, output_path, keep_zero=True):
    """
    Reorders all labels in a NIfTI image to a contiguous range starting from 1 (or 0 if keep_zero=True).
    Useful for cleaning up segmentation masks with disjointed or sparse label IDs.

    Parameters
    ----------
    nifti_image_path : str
        Path to the input NIfTI image with sparse or unordered labels.

    output_path : str
        Path where the reordered NIfTI image will be saved.

    keep_zero : bool, optional
        If True, label 0 will be preserved as background. If False, all labels start from 1.

    Returns
    -------
    dict
        A dictionary mapping original labels to new labels.
    """
    img = nib.load(nifti_image_path)
    data = img.get_fdata().astype(np.int32)

    unique_labels = np.unique(data)
    unique_labels = unique_labels[unique_labels !=
                                  0] if keep_zero else unique_labels

    label_mapping = {label: new_label for new_label, label in enumerate(
        sorted(unique_labels), start=1 if keep_zero else 0)}

    reordered_data = np.zeros_like(data, dtype=np.int32)
    for original, new in label_mapping.items():
        reordered_data[data == original] = new

    new_img = nib.Nifti1Image(reordered_data, img.affine, img.header)
    nib.save(new_img, output_path)

    return label_mapping
