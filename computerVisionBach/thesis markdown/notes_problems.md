### Preprocessing Bug Report: SegFormer vs. Mask2Former

#### Issue Summary

During training of the `nvidia/segformer-b2-finetuned-ade-512-512` model using Hugging Face's `AutoImageProcessor`, the model failed to learn effectively. Loss values showed minimal improvement, and evaluation metrics (e.g., mIoU) remained low throughout training.

---

#### Diagnosis

Upon inspecting the model inputs, it was found that the pixel value range of the input tensors was unexpectedly high:

```
tensor.min(): ~15.97  
tensor.max(): ~770.80
```

This indicated that image data (originally `uint8` in `[0, 255]`) had not been rescaled before normalization. As a result, ImageNet mean/std normalization was applied to raw pixel values, resulting in out-of-distribution inputs.

---

#### Root Cause

The image processor was instantiated using:

```python
processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
```

This implicitly used `do_rescale=False` and `do_normalize=True`, meaning:

* Pixel values were **not divided by 255**.
* ImageNet normalization (mean/std) was still applied.

Thus, normalization was incorrectly applied to `[0, 255]` data instead of `[0.0, 1.0]`, producing tensor values as high as \~770.

---

#### Impact on SegFormer

SegFormer uses a MixVision Transformer encoder pretrained on ImageNet, which expects:

* Inputs scaled to `[0.0, 1.0]`
* Then normalized using ImageNet `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`
* Fixed input size of `(512, 512)`

Because inputs were not rescaled, the model received out-of-distribution data, breaking the pretrained encoder's assumptions and leading to poor performance.

---

#### Why Mask2Former Still Worked

The Mask2Former model uses a Swin Transformer backbone, which is more tolerant to input variation due to:

* Pretraining on large datasets like COCO and ADE20K
* Support for variable image sizes
* More robust architecture design

As a result, even with suboptimal preprocessing, Mask2Former continued to learn effectively.

---

#### Solution

The issue was resolved by explicitly specifying the correct preprocessing parameters:

```python
processor = AutoImageProcessor.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    do_rescale=True,
    do_normalize=True,
    reduce_labels=False,
    size={"height": 512, "width": 512}
)
```

A post-processor diagnostic confirmed that input tensors were now in the correct range:

```
tensor.min(): ~-2.11  
tensor.max(): ~+2.62
```

SegFormer performance significantly improved after this fix.

---

#### Key Stuff:

When using pretrained segmentation models from Hugging Face, especially transformer-based models like SegFormer:

* Always pass raw `uint8` RGB images
* Set `do_rescale=True` and `do_normalize=True` explicitly
* Avoid manual normalization before the processor
* Match the expected input resolution (e.g., 512×512 for SegFormer)

This ensures that inputs match the distribution expected by the pretrained backbone and avoids silent degradation in model performance.
