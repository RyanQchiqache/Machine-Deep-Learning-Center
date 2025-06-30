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
