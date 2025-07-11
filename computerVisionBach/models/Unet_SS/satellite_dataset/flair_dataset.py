import os
import glob
import numpy as np
from computerVisionBach.models.Unet_SS.satellite_dataset.satellite_data import SatelliteDataset
from computerVisionBach.models.Unet_SS import utils

class FlairDataset(SatelliteDataset):
    COLOR_MAP = {
        1: (219, 14, 154),   # building
        2: (147, 142, 123),  # pervious surface
        3: (248, 12, 0),     # impervious surface
        4: (169, 113, 1),    # bare soil
        5: (21, 83, 174),    # water
        6: (25, 74, 38),     # coniferous
        7: (70, 228, 131),   # deciduous
        8: (243, 166, 13),   # brushwood
        9: (102, 0, 130),    # vineyard
        10: (85, 255, 0),    # herbaceous vegetation
        11: (255, 243, 13),  # agricultural land
        12: (228, 223, 124), # plowed land
        13: (61, 230, 235),  # swimming pool
        14: (255, 255, 255), # snow
        15: (138, 179, 160), # clear cut
        16: (107, 113, 79),  # mixed
        17: (197, 220, 66),  # ligneous
        18: (153, 153, 255), # greenhouse
        19: (0, 0, 0),       # other
    }

    COLOR_TO_CLASS = utils.create_color_to_class(COLOR_MAP)

    @staticmethod
    def rgb_to_class(mask_rgb: np.ndarray) -> np.ndarray:
        H, W, _ = mask_rgb.shape
        class_map = np.zeros((H, W), dtype=np.uint8)
        for color, class_idx in FlairDataset.COLOR_TO_CLASS.items():
            matches = np.all(mask_rgb == color, axis=-1)
            class_map[matches] = class_idx
        return class_map

    def __init__(self, image_input, mask_input, transform=None):
        if isinstance(image_input, (list, tuple, np.ndarray)) and isinstance(mask_input, (list, tuple, np.ndarray)):
            image_paths = image_input
            mask_paths = mask_input
        else:
            image_paths = sorted(glob.glob(os.path.join(image_input, "**", "*.tif"), recursive=True) +
                                 glob.glob(os.path.join(image_input, "**", "*.jpg"), recursive=True))
            mask_paths = sorted(glob.glob(os.path.join(mask_input, "**", "*.tif"), recursive=True) +
                                glob.glob(os.path.join(mask_input, "**", "*.png"), recursive=True))

        super().__init__(image_paths, mask_paths, transform=transform, rgb_to_class=FlairDataset.rgb_to_class)
