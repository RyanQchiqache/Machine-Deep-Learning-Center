import torch
from matplotlib import pyplot as plt
from computerVisionBach.models.Unet_SS import model_pipeline
import numpy as np


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
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs, features = model(images, return_features=True)
            preds = torch.argmax(outputs, dim=1)

            image_np = images[0].cpu().permute(1, 2, 0).numpy()
            mask_np = masks[0].cpu().numpy()
            pred_np = preds[0].cpu().numpy()

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1);
            plt.imshow(image_np);
            plt.title("Image")
            plt.subplot(1, 3, 2);
            plt.imshow(model_pipeline.class_to_rgb(mask_np));
            plt.title("Ground Truth")
            plt.subplot(1, 3, 3);
            plt.imshow(model_pipeline.class_to_rgb(pred_np));
            plt.title("Prediction")
            plt.tight_layout();
            plt.show()

            for layer_name in ["bottle_neck", "enc1", "enc2", "dec3", "dec4"]:
                visualise_feture_map(features[layer_name], "feature channels")
            break


#=======================================
# Visualise feature map
#=======================================
def visualise_feture_map(feature_map, title):
    if feature_map.dim() == 4:
        feature_map = feature_map[0]

    num_channels = feature_map.shape[0]
    plt.figure(figsize=(15, 15))
    for i in range(min(num_channels, 16)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(feature_map[i].detach().cpu().numpy(), cmap="viridis")
        plt.axis("off")
        plt.title(f"Channel{i}")
    plt.title(title)
    plt.tight_layout()
    plt.show()
