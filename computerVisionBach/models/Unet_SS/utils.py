from collections import Counter

import numpy as np
from omegaconf import OmegaConf
cfg = OmegaConf.load("config.yaml")

def rgb_to_class_label(mask, color_map):
    label = np.full(mask.shape[:2], fill_value=255, dtype=np.uint8)  # invalid default
    for class_id, rgb in color_map.items():
        match = np.all(mask == rgb, axis=-1)
        label[match] = class_id
    if (label == 255).any():
        unique_unmapped = np.unique(mask[label == 255])
        print(f"⚠️ Warning: Unmapped RGB values found: {unique_unmapped}")
        raise ValueError("Some pixels in mask have no class mapping.")
    return label

def relabel_mask(mask: np.ndarray, original_labels: list) -> np.ndarray:
    """
    Remaps original sparse label values (e.g., [1, 2, 6, 18]) to contiguous [0, 1, 2, ..., N-1]
    so CrossEntropyLoss works without index errors.
    """
    label_map = {orig: new for new, orig in enumerate(sorted(original_labels))}
    remapped = np.zeros_like(mask)
    for orig_label, new_label in label_map.items():
        remapped[mask == orig_label] = new_label
    return remapped



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

def count_class_distribution(masks):
    total_counts = Counter()
    for mask in masks:
        values, counts = np.unique(mask, return_counts=True)
        total_counts.update(dict(zip(values, counts)))
    return total_counts


def create_color_to_class(label_dict):
    """
    Create a reverse mapping from RGB color to class index.

    Example:
        Input:  {1: (219, 14, 154), 2: (147, 142, 123)}
        Output: {(219, 14, 154): 1, (147, 142, 123): 2}
    """
    if label_dict is None:
        raise ValueError("label_dict must be provided to create_color_to_class")

    color_to_class = {color: class_index for class_index, color in label_dict.items()}
    return color_to_class

CLASS_COLOR_MAP = {
    0: np.array([60, 16, 152]),
    1: np.array([132, 41, 246]),
    2: np.array([110, 193, 228]),
    3: np.array([254, 221, 58]),
    4: np.array([226, 169, 41]),
    5: np.array([155, 155, 155])
}
COLOR_MAP_dense = {
    0: ['Low vegetation', '#f423e8'],
    1: ['Paved road', '#66669c'],
    2: ['Non paved road', '#be9999'],
    3: ['Paved parking place', '#999999'],
    4: ['Non paved parking place', '#faaa1e'],
    5: ['Bikeways', '#98fb98'],
    6: ['Sidewalks', '#4682b4'],
    7: ['Entrance exit', '#6b8e23'],
    8: ['Danger area', '#dcdc00'],
    9: ['Lane-markings', '#ff0000'],
    10: ['Building', '#dc143c'],
    11: ['Car', '#7d008e'],
    12: ['Trailer', '#aac828'],
    13: ['Van', '#c83c64'],
    14: ['Truck', '#961250'],
    15: ['Long truck', '#51b451'],
    16: ['Bus', '#bef115'],
    17: ['Clutter', '#0b7720'],
    18: ['Impervious surface', '#78f078'],
    19: ['Tree', '#464646'],
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


class_names = [
    "Low vegetation",           # class 1
    "Paved road",               # class 2
    "Non paved road",           # class 3
    "Paved parking place",      # class 4
    "Non paved parking place",  # class 5
    "Bikeways",                 # class 6
    "Sidewalks",                # class 7
    "Entrance exit",            # class 8
    "Danger area",              # class 9
    "Lane-markings",            # class 10
    "Building",                 # class 11
    "Car",                      # class 12
    "Trailer",                  # class 13
    "Van",                      # class 14
    "Truck",                    # class 15
    "Long truck",               # class 16
    "Bus",                      # class 17
    "Clutter",                  # class 18
    "Impervious surface",       # class 19
    "Tree",                     # class 20
]
import csv, os

def append_results_csv(csv_path, row_dict):
    # Creates header if file doesn’t exist yet
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if new_file:
            writer.writeheader()
        writer.writerow(row_dict)

row = {
    "dataset": "FLAIR",
    "split": "test",
    "model": model_name,                 # e.g. "Unet++"
    "encoder": encoder_name,             # e.g. "resnet50"
    "seed": seed,
    **test_metrics                       # mIoU_macro, OA_micro, OA_macro, BoundaryF1, mIoU_rare, Params_M, PeakVRAM_GB, Latency_ms_per_img
}
append_results_csv("results_runs.csv", row)

# ================================
# Reconstruction of images
# ================================
def reconstruct_two_examples(model, test_dataset, color_map, num_reconstructions):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    patch_size = 512
    patch_shape = (4, 4)
    patches_per_image = patch_shape[0] * patch_shape[1]

    all_images, all_masks, all_preds = [], [], []
    count = 0

    with torch.no_grad():
        for img_tensor, mask_tensor in test_loader:
            if count >= num_reconstructions:
                break

            img_tensor = img_tensor.cuda()
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            all_images.append(img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
            all_masks.append(mask_tensor.squeeze(0).numpy())
            all_preds.append(pred)

            # Once 16 patches collected, reconstruct a full image
            if len(all_images) == patches_per_image:
                save_path = f"reconstructed_outputs/reconstruction_{count}.png"
                visualisation.reconstruct_and_visualize_patches(
                    np.array(all_images), np.array(all_masks), np.array(all_preds),
                    patch_size=patch_size,
                    grid_shape=patch_shape,
                    color_map=color_map,
                    save_path=save_path
                )
                count += 1
                all_images, all_masks, all_preds = [], [], []

"""# ================================
# patchify and load data kaggel
# ================================
def extract_patches_from_directory(directory, kind='images'):
    dataset = []
    for path, subdirs, files in os.walk(directory):
        if path.endswith(kind):
            for file in sorted(os.listdir(path)):
                if (kind == 'images' and file.endswith('.jpg')) or (kind == 'masks' and file.endswith('.png')):
                    img_path = os.path.join(path, file)
                    img = cv2.imread(img_path, 1)
                    if kind == 'masks':
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w = (img.shape[0] // PATCH_SIZE) * PATCH_SIZE, (img.shape[1] // PATCH_SIZE) * PATCH_SIZE
                    img = Image.fromarray(img).crop((0, 0, w, h))
                    img = np.array(img)
                    patches = patchify(img, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_SIZE)
                    for i in range(patches.shape[0]):
                        for j in range(patches.shape[1]):
                            dataset.append(patches[i, j, 0])
    return np.array(dataset)


def load_data(root_dir, test_size, seed):
    images = extract_patches_from_directory(root_dir, kind='images')
    masks_rgb = extract_patches_from_directory(root_dir, kind='masks')
    masks_label = utils.convert_masks_to_class_labels(masks_rgb)

    visualisation.visualize_sample(images, masks_rgb, masks_label)

    X_train, X_test, y_train, y_test = train_test_split(images, masks_label, train_size=1 - test_size,
                                                        random_state=seed)
    train_dataset = SatelliteDataset(X_train, y_train)
    test_dataset = SatelliteDataset(X_test, y_test)

    return train_dataset, test_dataset"""


# ================================
# Evaluation
# ================================
def evaluate(model, dataloader, device, epoch=None, writer=None):
    model.eval()
    iou_macro = MulticlassJaccardIndex(num_classes=N_CLASSES, average='macro').to(device)
    iou_per_class = MulticlassJaccardIndex(num_classes=N_CLASSES, average=None).to(device)
    accuracy = MulticlassAccuracy(num_classes=N_CLASSES, average='macro').to(device)
    is_mask2former = isinstance(model, Mask2FormerForUniversalSegmentation)
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            # HuggingFace transformer models
            """if isinstance(model, PreTrainedModel):
                images_np = [img.permute(1, 2, 0).cpu().numpy() for img in images]
                inputs = processor(images=images_np, return_tensors="pt", do_rescale=False).to(device)
                output = model(**inputs)
                logits = output.logits
            else:"""
            if is_mask2former:
                batch_inputs = processor(images=images, return_tensors="pt")
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                logits = model(**batch_inputs)
            else:
                logits = model(images)


            # Resize logits if needed
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )

            preds = torch.argmax(logits, dim=1)
            preds = preds.view(-1)
            masks = masks.view(-1)

            valid = masks != 255
            preds = preds[valid]
            masks = masks[valid]

            assert preds.min() >= 0 and preds.max() < N_CLASSES, f"Bad preds: {preds.unique()}"
            assert masks.min() >= 0 and masks.max() < N_CLASSES, f"Bad masks: {masks.unique()}"

            # Incremental metric updates (no memory explosion)
            iou_macro.update(preds, masks)
            iou_per_class.update(preds, masks)
            accuracy.update(preds, masks)

    # Compute final metrics
    miou = iou_macro.compute()
    per_class_ious = iou_per_class.compute()
    acc = accuracy.compute()

    logger.info(f"\n✓ Mean IoU: {miou:.4f}")
    for i, class_iou in enumerate(per_class_ious):
        logger.info(f"  └─ Class {i:02d} IoU: {class_iou:.4f}")

    # Write to TensorBoard if available
    if writer and epoch is not None:
        writer.add_scalar("IoU/Mean", miou, epoch)
        writer.add_scalar("Accuracy/Mean", acc, epoch)
        for i, val in enumerate(per_class_ious):
            writer.add_scalar(f"IoU/Class_{i}", val, epoch)

    # Reset metrics after evaluation (optional if reused)
    iou_macro.reset()
    iou_per_class.reset()

    return miou