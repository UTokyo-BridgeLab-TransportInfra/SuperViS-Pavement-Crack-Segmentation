# Super-Visual Segmentation (SuperViS) using low-quality mobile imagery
This repository defines a general-domain problem: whether low-visibility images still contain latent feature information when direct human annotation is no longer viable, whether such images can still be leveraged for downstream segmentation, and how these features can be labelled accurately at low cost. Using degraded in-vehicle mobile video of pavement cracks as a challenging case, we find that more than half of the frames are degraded to the extent that cracks are not readily observable with the human vision. To address this, we developed Super-Visual Segmentation (SuperViS) using low-quality mobile imagery. SuperViS leverages spatially-aligned clear–degraded image pairs captured at the same road locations to learn dense correspondence and transfer supervision from clear images to their degraded counterparts. Using a large-scale dataset (600k crack images collected across Japan and China) containing substantial real-world degradation, we show that low visibility does not imply data deficiency: even if accuracy is below specialised sensors, consumer-grade degraded video can still support effective initial screening.

<img width="922" height="341" alt="image" src="https://github.com/user-attachments/assets/d9d279d6-b7cb-4074-b8cb-7cf6c482de2e" />

**Figure 1**. Using the proposed annotation framework, a single ResUNet model achieves segmentation performance on smartphone imagery that is comparable to expert annotations from a line-scan camera inspection vehicle, even under backlighting, deep shadows and high-speed motion. Notably, in regions where cracks are almost imperceptible in degraded images, the model predictions remain consistent with annotations derived from the corresponding clear images.

## Quick Start: Minimal Training Demo (3-Stage SuperVis Pipeline)

This repository provides a minimal dataset and scripts to reproduce the proposed three-stage pipeline:
1) cross-condition matching,  
2) human-in-the-loop weak supervision on clear images,  
3) degraded-image crack segmentation without manual annotation.

> **Note**: Commands below are written for terminal usage. If you run them inside a Jupyter notebook, prefix with `!`.

---

### Stage 1 — Cross-condition Matching (clear–degraded alignment)

```bash
cd "Cross-condition Matching"
python train.py
```

This step trains the image-matching model used to align clear and degraded images and estimate correspondences.

---

### Stage 2 — Human-in-the-loop Weak Supervision on Clear Images

```bash
cd "../Human-in-the-loop Weak Supervision on Clear Images"

python train.py \
  -data_dir dataset/train/ \
  -ENCODER resnet101 \
  -model_dir weights1/ \
  -augment \
  -rect
```

After training is completed, run crack-width restoration and post-processing:

```bash
jupyter notebook demo.ipynb
```

Open `demo.ipynb` and **run all cells** to perform crack width restoration and additional post-processing steps.

---

### Stage 3 — Degraded-image Crack Segmentation without Human Annotation

```bash
cd "../Degraded image crack segmentation without human annotation"
```

#### (Optional) Preprocessing
Generate intermediate data required for training:

```bash
jupyter notebook preprocessing.ipynb
```

> If you use the provided **minimal training demonstration dataset**, you can **skip** this preprocessing step and proceed directly to training.

#### Train the degraded-image segmentation model

```bash
python train.py \
  -data_dir dataset/training_dataset/train/ \
  -pretrained pretrained_weights/pretrained_weights_from_clear_image.pt \
  -ENCODER resnet101 \
  -n_epoch 10 \
  -batch_size 32 \
  -augment \
  -rect_training
```

**Important**: During training, the directory  
`dataset/training_dataset/train/masks/train`  
is continuously updated and overwritten with refined pseudo-labels.

#### Restore updated pseudo-labels (post-processing)

```bash
jupyter notebook "Training and post-processing.ipynb"
```

Open `Training and post-processing.ipynb` and run all cells to restore the updated pseudo-labels and apply the final post-processing steps.

**Result**

&nbsp;&nbsp;For reference and reproducibility, the corresponding outputs of the matching and segmentation models used in this study are publicly released.

**Dataset**

&nbsp;&nbsp;The repository currently provides only sample data for demonstration. Upon acceptance of the paper, we will release a portion of the datasets used in this study.
