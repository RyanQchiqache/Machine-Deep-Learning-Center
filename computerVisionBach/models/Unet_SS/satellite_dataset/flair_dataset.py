import os
import glob
import numpy as np
from computerVisionBach.models.Unet_SS.satellite_dataset.satellite_data import SatelliteDataset
from computerVisionBach.models.Unet_SS import utils

class FlairDataset(SatelliteDataset):
    COLOR_MAP = {
        0: (219, 14, 154),   # building
        1: (147, 142, 123),  # pervious surface
        2: (248, 12, 0),     # impervious surface
        3: (169, 113, 1),    # bare soil
        4: (21, 83, 174),    # water
        5: (25, 74, 38),     # coniferous
        6: (70, 228, 131),   # deciduous
        7: (243, 166, 13),   # brushwood
        8: (102, 0, 130),    # vineyard
        9: (85, 255, 0),    # herbaceous vegetation
        10: (255, 243, 13),  # agricultural land
        11: (228, 223, 124), # plowed land
        12: (61, 230, 235),  # swimming pool
        13: (255, 255, 255), # snow
        14: (138, 179, 160), # clear cut
        15: (107, 113, 79),  # mixed
        16: (197, 220, 66),  # ligneous
        17: (153, 153, 255), # greenhouse
        18: (0, 0, 0),       # other
    }

    COLOR_TO_CLASS = utils.create_color_to_class(COLOR_MAP)
    def __init__(self, image_input, mask_input, transform=None, relabel_fn=None, allowed_labels=None, use_processor=None, is_hf_model=None):
        if isinstance(image_input, (list, tuple, np.ndarray)) and isinstance(mask_input, (list, tuple, np.ndarray)):
            image_paths = image_input
            mask_paths = mask_input
        else:
            image_paths = sorted(glob.glob(os.path.join(image_input, "**", "*.tif"), recursive=True) +
                                 glob.glob(os.path.join(image_input, "**", "*.jpg"), recursive=True))
            mask_paths = sorted(glob.glob(os.path.join(mask_input, "**", "*.tif"), recursive=True) +
                                glob.glob(os.path.join(mask_input, "**", "*.png"), recursive=True))

        super().__init__(
            images=image_paths,
            masks=mask_paths,
            transform=transform,
            relabel_fn=relabel_fn,
            allowed_labels = allowed_labels,
            use_processor = use_processor,
            is_hf_model = is_hf_model
        )

