import numpy as np

def rgb_to_class_label(mask, color_map):
    label = np.zeros(mask.shape[:2], dtype=np.uint8)
    for class_id, rgb in color_map.items():
        label[np.all(mask == rgb, axis=-1)] = class_id
    return label


def convert_masks_to_class_labels(masks):
    return np.array([rgb_to_class_label(mask) for mask in masks])


def class_to_rgb(mask, color_map):
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        rgb_mask[mask == class_id] = color
    return rgb_mask

def remap_mask(mask):
    result = np.zeros_like(mask, dtype=np.uint8)
    for i, class_id in enumerate(sorted(np.unique(mask))):
        result[mask == class_id] = i
    return result

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

CLASS_COLOR_MAP = {
    0: np.array([60, 16, 152]),
    1: np.array([132, 41, 246]),
    2: np.array([110, 193, 228]),
    3: np.array([254, 221, 58]),
    4: np.array([226, 169, 41]),
    5: np.array([155, 155, 155])
}
COLOR_MAP_dense = {
    1: ['Low vegetation', '#f423e8'],
    2: ['Paved road', '#66669c'],
    3: ['Non paved road', '#be9999'],
    4: ['Paved parking place', '#999999'],
    5: ['Non paved parking place', '#faaa1e'],
    6: ['Bikeways', '#98fb98'],
    7: ['Sidewalks', '#4682b4'],
    8: ['Entrance exit', '#6b8e23'],
    9: ['Danger area', '#dcdc00'],
    10: ['Lane-markings', '#ff0000'],
    11: ['Building', '#dc143c'],
    12: ['Car', '#7d008e'],
    13: ['Trailer', '#aac828'],
    14: ['Van', '#c83c64'],
    15: ['Truck', '#961250'],
    16: ['Long truck', '#51b451'],
    17: ['Bus', '#bef115'],
    18: ['Clutter', '#0b7720'],
    19: ['Impervious surface', '#78f078'],
    20: ['Tree', '#464646'],
}
COLOR_MAP_multi_lane = {
    0: ['Background', '#000000'],
    1: ['Dash Line', '#ff0000'],
    2: ['Long Line', '#0000ff'],
    3: ['Small dash line', '#ffff00'],
    4: ['Turn signs', '#00ff00'],
    5: ['Other signs', '#ff8000'],
    6: ['Plus sign on crossroads', '#800000'],
    7: ['Crosswalk', '#00ffff'],
    8: ['Stop line', '#008000'],
    9: ['Zebra zone', '#ff00ff'],
    10: ['No parking zone', '#009696'],
    11: ['Parking space', '#c8c800'],
    12: ['Other lane-markings', '#6400c8'],
    }

CLASS_COLOR_MAP = {
    0: np.array([60, 16, 152]),
    1: np.array([132, 41, 246]),
    2: np.array([110, 193, 228]),
    3: np.array([254, 221, 58]),
    4: np.array([226, 169, 41]),
    5: np.array([155, 155, 155])
}