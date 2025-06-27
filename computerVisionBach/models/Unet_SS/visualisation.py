import torch
from matplotlib import pyplot as plt
from computerVisionBach.models.Unet_SS import utils
import numpy as np
from computerVisionBach.models.Unet_SS.model_pipeline import COLOR_MAP_dense, COLOR_MAP_multi_lane

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
