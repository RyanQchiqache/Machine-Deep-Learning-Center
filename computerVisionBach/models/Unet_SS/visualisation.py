import torch
import numpy as np
from computerVisionBach.models.Unet_SS import utils
from computerVisionBach.models.Unet_SS.utils import COLOR_MAP_dense, COLOR_MAP_multi_lane
from matplotlib import pyplot as plt
from patchify import unpatchify
import os


color_map = {k: utils.hex_to_rgb(v[1]) for k, v in COLOR_MAP_dense.items()}
# ================================
# Prediction Sample
# ================================
def visualize_sample(images, masks, labels):
    idx = np.random.randint(0, len(images) - 1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(images[idx])
    plt.title("Image")
    plt.subplot(1, 3, 2)
    plt.imshow(masks[idx])
    plt.title("RGB Mask")
    plt.subplot(1, 3, 3)
    plt.imshow(labels[idx])
    plt.title("Label Mask")
    plt.tight_layout()
    plt.show()

# ================================
# Prediction Visualization
# ================================
def visualize_prediction(model, dataloader, device):
    model.eval()
    shown = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            batch_size = images.shape[0]
            for i in range(batch_size):
                if shown >= 10:
                    return

                image_np = images[i].cpu().permute(1, 2, 0).numpy()
                mask_np = masks[i].cpu().numpy()
                pred_np = preds[i].cpu().numpy()

                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(image_np)
                plt.title(f"Image {shown+1}")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(utils.class_to_rgb(mask_np, color_map))
                plt.title("Ground Truth")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(utils.class_to_rgb(pred_np, color_map))
                plt.title("Prediction")
                plt.axis("off")

                plt.tight_layout()
                plt.show()

                shown += 1


            """for layer_name in ["bottleneck", "enc1", "enc2", "dec3", "dec4"]:
                visualise_feature_map(features[layer_name], "feature channels")
            break
            """

#=======================================
# Visualise feature map
#=======================================
def visualise_feature_map(feature_map, title):
    if feature_map.dim() == 4:
        feature_map = feature_map[0]

    num_channels = feature_map.shape[0]
    plt.figure(figsize=(18, 18))
    for i in range(min(num_channels, 12)):
        plt.subplot(3, 4, i + 1)
        plt.imshow(feature_map[i].detach().cpu().numpy(), cmap="viridis")
        plt.axis("off")
        plt.title(f"Channel{i}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def reconstruct_and_visualize_patches(image_patches, mask_patches, pred_patches, patch_size, grid_shape, color_map, save_path=None):
    """
    image_patches, mask_patches, pred_patches: np.array of shape (n_patches, H, W, C) or (n_patches, H, W)
    grid_shape: tuple of (rows, cols)
    """
    assert len(image_patches) == grid_shape[0] * grid_shape[1], "Patch count doesn't match grid shape"

    h, w = patch_size, patch_size
    img_array = image_patches.reshape(grid_shape[0], grid_shape[1], h, w, 3)
    mask_array = mask_patches.reshape(grid_shape[0], grid_shape[1], h, w)
    pred_array = pred_patches.reshape(grid_shape[0], grid_shape[1], h, w)

    full_image = unpatchify(img_array, (grid_shape[0] * h, grid_shape[1] * w, 3))
    full_mask = unpatchify(mask_array, (grid_shape[0] * h, grid_shape[1] * w))
    full_pred = unpatchify(pred_array, (grid_shape[0] * h, grid_shape[1] * w))

    full_mask_rgb = utils.class_to_rgb(full_mask, color_map)
    full_pred_rgb = utils.class_to_rgb(full_pred, color_map)

    # Visualization
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(full_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(full_mask_rgb)
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(full_pred_rgb)
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()

    plt.close()
