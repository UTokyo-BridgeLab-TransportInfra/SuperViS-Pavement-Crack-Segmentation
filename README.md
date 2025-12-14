# nagayamalab-crack-segmentation

This repository provides an end-to-end framework for pavement crack segmentation under extreme low-visibility conditions (backlighting, deep shadows, defocus and motion blur) in vehicle-mounted smartphone videos, without any image restoration stage. We leverage cross-condition supervision: clear views supervise the corresponding degraded views. As a result, even in severely degraded regions where cracks are barely discernible to human annotators, our method can recover most of the crack structures.

<img width="922" height="341" alt="image" src="https://github.com/user-attachments/assets/d9d279d6-b7cb-4074-b8cb-7cf6c482de2e" />

Figure 1. Using the proposed annotation framework, a single ResUNet model achieves segmentation performance on smartphone imagery that is comparable to expert annotations from a line-scan camera inspection vehicle, even under backlighting, deep shadows and high-speed motion. Notably, in regions where cracks are almost imperceptible in degraded images, the model predictions remain consistent with annotations derived from the corresponding clear images.

Our codebase is organized into two main modules:

1) Cross-condition Matching (Clear ↔ Degraded)
Cross-condition matching is built upon a modified LoFTR model tailored for pavement imagery. The matcher is optimized for low-texture road surfaces and strong appearance changes, allowing robust clear–degraded correspondence estimation and reliable crack region localization in challenging smartphone videos.

3) Crack Annotation & Pseudo-label Refinement
We provide a practical annotation pipeline that enables supervision even when cracks are only weakly visible, by transferring reliable labels from clear images to corresponding degraded views via matching-based localization and refining them into accurate pseudo-labels through iterative, rule-based propagation that is robust to residual misalignment and local distortions.
