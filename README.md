# Nagayama-lab crack segmentation
This repository provides an end-to-end framework for pavement crack segmentation under extreme low-visibility conditions (backlighting, deep shadows, defocus and motion blur) in vehicle-mounted smartphone videos, without any image restoration stage. We leverage cross-condition supervision: clear views supervise the corresponding degraded views. As a result, even in severely degraded regions where cracks are barely discernible to human annotators, our method can recover most of the crack structures.

<img width="922" height="341" alt="image" src="https://github.com/user-attachments/assets/d9d279d6-b7cb-4074-b8cb-7cf6c482de2e" />

**Figure 1**. Using the proposed annotation framework, a single ResUNet model achieves segmentation performance on smartphone imagery that is comparable to expert annotations from a line-scan camera inspection vehicle, even under backlighting, deep shadows and high-speed motion. Notably, in regions where cracks are almost imperceptible in degraded images, the model predictions remain consistent with annotations derived from the corresponding clear images.

Our codebase is organized into two main modules:

**1) Cross-condition Matching (Clear ↔ Degraded)**
   
&nbsp;&nbsp;Cross-condition matching is built upon a modified LoFTR model tailored for pavement imagery. The matcher is optimized for low-texture road surfaces and strong appearance changes, allowing robust clear–degraded correspondence estimation and reliable crack region localization in challenging smartphone videos.

**2) Crack Annotation & Pseudo-label Refinement**

&nbsp;&nbsp;We provide a practical annotation pipeline that enables supervision even when cracks are only weakly visible, by transferring reliable labels from clear images to corresponding degraded views via matching-based localization and refining them into accurate pseudo-labels through iterative, rule-based propagation that is robust to residual misalignment and local distortions.

**Result**

&nbsp;&nbsp;For reference and reproducibility, the corresponding outputs of the matching and segmentation models used in this study are publicly released.

**Dataset**

&nbsp;&nbsp;Upon acceptance of the paper, we will release all validation datasets used in this study. The release will include (i) a matched pavement image dataset of approximately 13k image pairs, where each image covers an area of roughly 4 m × 6 m of roadway, spanning diverse road materials, illumination conditions and stages of deterioration; and (ii) a 40k-image crack segmentation dataset annotated using the proposed framework, where each image corresponds to approximately 4–7 m² of pavement surface. Of these, 20k images represent degraded views in which cracks are difficult to discern over most regions. Each degraded image is paired with a corresponding clear image and its annotation. To the best of our knowledge, this constitutes the first large-scale dataset for road image matching and crack segmentation collected under real-world degraded conditions.
