import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_li
from skimage.measure import label, regionprops
from skimage.morphology import disk, opening, remove_small_objects
from skimage.segmentation import watershed


def get_binary_map(img, mode):
    # 定义不同模式对应的 RGB 通道权重系数
    mode_weights = {
        'r': np.array([2, 0, 0]),
        'g': np.array([0, 2, 0]),
        'b': np.array([0, 0, 2]),
        # 可以根据需要添加更多模式及对应权重
    }
    if mode not in mode_weights:
        raise ValueError(f"不支持的模式 {mode}，请选择合适的模式。")

    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)

    sub_rgb = (r ** mode_weights[mode][0] + g ** mode_weights[mode][1] + b ** mode_weights[mode][2]).astype(np.float32)
    thresh = threshold_li(sub_rgb)
    binary_map = sub_rgb > thresh

    return binary_map.astype(bool)


def apply_opening(binary_img, selem_parameter, remove_objects):
    selem = disk(selem_parameter)
    opened_image = opening(binary_img, selem)
    opened_image = remove_small_objects(opened_image.astype(bool), remove_objects).astype(np.int64)

    return opened_image


def find_median_cell_size(labeled_img):
    area = []
    for region in regionprops(labeled_img):
        area.append(region.area)
    median = np.median(np.array(area))
    return median


def apply_watershed(labeled_img, median_size, min_distance=40, remove_objects=1000):
    if labeled_img.size == 0:
        return labeled_img

    bool_labeled_img = labeled_img.astype(bool)
    distance = ndi.distance_transform_edt(bool_labeled_img)
    local_max_coords = peak_local_max(distance, min_distance=min_distance, exclude_border=0)
    if local_max_coords.size == 0:
        print("未找到局部最大值点，直接返回原始图像")
        return labeled_img

    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = label(local_max_mask)
    labels = watershed(-distance, markers, mask=bool_labeled_img, watershed_line=True)

    large_cells = remove_small_objects(bool_labeled_img, 2 * median_size).astype(np.int64)
    final = labeled_img
    final[large_cells > 0] = labels[large_cells > 0]

    labeled_final = label(final)
    final = remove_small_objects(labeled_final, remove_objects).astype(np.int64)

    return final
