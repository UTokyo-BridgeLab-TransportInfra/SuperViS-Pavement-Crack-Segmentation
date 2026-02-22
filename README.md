# Super-Visual Segmentation (SuperViS) using low-quality mobile imagery
This repository defines a general-domain problem: whether low-visibility images still contain latent feature information when direct human annotation is no longer viable, whether such images can still be leveraged for downstream segmentation, and how these features can be labelled accurately at low cost. Using degraded in-vehicle mobile video of pavement cracks as a challenging case, we find that more than half of the frames are degraded to the extent that cracks are not readily observable with the human vision. To address this, we developed Super-Visual Segmentation (SuperViS) using low-quality mobile imagery. SuperViS leverages spatially-aligned clear–degraded image pairs captured at the same road locations to learn dense correspondence and transfer supervision from clear images to their degraded counterparts. Using a large-scale dataset (600k crack images collected across Japan and China) containing substantial real-world degradation, we show that low visibility does not imply data deficiency: even if accuracy is below specialised sensors, consumer-grade degraded video can still support effective initial screening.

<img width="922" height="341" alt="image" src="https://github.com/user-attachments/assets/d9d279d6-b7cb-4074-b8cb-7cf6c482de2e" />

**Figure 1**. Using the proposed annotation framework, a single ResUNet model achieves segmentation performance on smartphone imagery that is comparable to expert annotations from a line-scan camera inspection vehicle, even under backlighting, deep shadows and high-speed motion. Notably, in regions where cracks are almost imperceptible in degraded images, the model predictions remain consistent with annotations derived from the corresponding clear images.

Our codebase is organized into two main modules:

**1) Cross-condition Matching (Clear ↔ Degraded)**
   
&nbsp;&nbsp;Cross-condition matching is built upon a modified LoFTR model tailored for pavement imagery. The matcher is optimized for low-texture road surfaces and strong appearance changes, allowing robust clear–degraded correspondence estimation and reliable crack region localization in challenging smartphone videos.

**2) Human-in-the-loop weak supervision on clear images using efficient 1-pixel centreline annotation**

&nbsp;&nbsp;To annotate dense cracks on clear images efficiently, we draw 1-pixel crack centrelines and use a watershed-based weak-supervision procedure to recover crack width and reduce the impact of inaccurate crack-width labels.

# Step 1: Enter the training directory
%cd Human-in-the-loop_weak_supervision_on_clear_images

# Step 2: Train the model
!python train.py \
    -data_dir dataset/train/ \
    -ENCODER resnet101 \
    -model_dir weights1/ \
    -augment \
    -rect

# Step 3: After training, open demo.ipynb
Run all cells in demo.ipynb to perform crack width restoration and additional post-processing steps.

**3) Degraded image crack segmentation without human annotation**

&nbsp;&nbsp;We provide a practical annotation pipeline that enables supervision even when cracks are only weakly visible, by transferring reliable labels from clear images to corresponding degraded views via matching-based localization and refining them into accurate pseudo-labels through iterative, rule-based propagation that is robust to residual misalignment and local distortions.

**Result**

&nbsp;&nbsp;For reference and reproducibility, the corresponding outputs of the matching and segmentation models used in this study are publicly released.

**Dataset**

&nbsp;&nbsp;The repository currently provides only sample data for demonstration. Upon acceptance of the paper, we will release a portion of the datasets used in this study.
