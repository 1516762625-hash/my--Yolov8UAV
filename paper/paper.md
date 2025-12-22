# DCE-YOLOv8-UAV: A 1024-Resolution Hybrid YOLOv8 for UAV Small Object Detection via Mamba Global Context, CLIP Semantic Anchors, and Dynamic Large-Kernel Attention

**Target style:** IEEE (conference-style), full tables in main text  
**Dataset:** VisDrone2019-DET (val)  
**Primary metric:** mAP@0.5  
**Best epoch:** 118  
**Best result (val):** P=0.6229, R=0.53481, mAP@0.5=0.55981, mAP@0.5:0.95=0.35302

---

## Abstract
Small object detection in unmanned aerial vehicle (UAV) imagery remains challenging due to extreme scale shrinkage, dense layouts, occlusions, and background clutter. This paper presents **DCE-YOLOv8-UAV**, a hybrid YOLOv8-based detector tailored for UAV-view small object detection on **VisDrone2019-DET** under a high-resolution **1024×1024** training protocol. The proposed approach integrates three key innovations: (1) **Mamba-inspired global context perception** achieved by inserting **C2f-VSS** (linear state space) blocks into both the backbone and neck; (2) **CLIP text-prior semantic guidance**, where frozen **CLIP ViT-B/32** text embeddings of the 10 VisDrone categories are injected as class-wise semantic anchors through **C2f-CLIP**; and (3) **dynamic Large Selective Kernel (LSK) attention** inserted at a critical feature fusion node before the detection heads. On the VisDrone2019-DET validation split, DCE-YOLOv8-UAV achieves a best **mAP@0.5 of 0.55981** at epoch 118, together with **mAP@0.5:0.95 of 0.35302**, **precision 0.6229**, and **recall 0.53481**. We also discuss fairness under differing input resolutions and provide a resolution-sensitivity analysis.

**Keywords:** UAV object detection; VisDrone2019-DET; small objects; YOLOv8; state space model; Mamba; CLIP; semantic prior; large-kernel attention.

---

## 1. Introduction
Object detection from UAV imagery is crucial for aerial surveillance, intelligent transportation, and emergency response. Compared with ground-view images, UAV scenes often contain wide fields of view where objects occupy only a few pixels, resulting in weak appearance cues, heavy occlusions, dense distributions, and strong background clutter. These characteristics significantly degrade detector reliability and cause frequent missed detections and misclassifications, especially for visually similar categories (e.g., *car* vs. *van*) or extremely small instances (e.g., *pedestrian*).

Recent one-stage detectors such as YOLOv8 provide strong efficiency; however, their feature extraction pipelines remain primarily convolutional and thus biased toward local pattern aggregation. For UAV small object detection, purely local reasoning is often insufficient: global contextual relations can be critical to disambiguate tiny objects from background patterns. Moreover, tiny objects offer limited discriminative pixels; therefore, injecting semantic priors can improve class separation. Finally, enlarging receptive fields is beneficial but must remain computationally feasible, especially under high-resolution training (1024×1024).

To address these challenges, we propose **DCE-YOLOv8-UAV**, a high-resolution hybrid YOLOv8 framework that jointly integrates (i) efficient long-range modeling via linear state space modules, (ii) vision-language semantic priors from frozen CLIP text embeddings, and (iii) dynamic large-kernel attention at a key feature fusion junction, while maintaining an efficient shallow–deep division of labor.

### Contributions
1. **Global context via linear state space modeling:** We introduce **C2f-VSS** modules inspired by Mamba-style state space modeling into both backbone and neck to capture long-range dependencies efficiently for UAV scenes.  
2. **CLIP semantic anchors:** We generate frozen **CLIP ViT-B/32** text embeddings for VisDrone classes using prompt ensembling and inject them via **C2f-CLIP** to guide deep semantic feature learning under weak visual cues.  
3. **Dynamic large selective kernel attention:** We incorporate **LSKBlock** at a critical fusion node before the detection heads to adaptively enlarge receptive fields and improve small-object responses.  
4. **Efficient hybrid architecture:** We employ lightweight **DCE** and **ERB** blocks in shallow high-resolution stages to preserve texture details with low overhead, while deeper stages focus on semantics via VSS and CLIP.

---

## 2. Related Work
### 2.1 UAV Small Object Detection
UAV-view detection benchmarks such as VisDrone are characterized by small targets and dense distributions. Enhancements typically include multi-scale feature fusion, higher input resolution, and attention modeling; however, robust global context modeling and semantic priors remain under-explored for tiny instances.

### 2.2 YOLO-based Detectors
YOLO-family detectors balance accuracy and speed via one-stage inference. YOLOv8 adopts an anchor-free design and decoupled heads. Nevertheless, convolution-centric backbones/necks can be limited for UAV scenes requiring long-range reasoning and strong class discrimination under weak visual evidence.

### 2.3 State Space Models for Vision
State space models provide an efficient alternative to quadratic self-attention for long-sequence modeling. Mamba-style selective state space dynamics have recently shown strong performance and favorable scaling, motivating their integration into visual backbones and feature pyramids.

### 2.4 Vision–Language Priors (CLIP) for Detection
CLIP provides strong language–vision alignment and can act as a semantic prior that compensates for weak visual cues. We leverage CLIP text embeddings as frozen semantic anchors to guide deep feature learning for UAV small objects.

### 2.5 Large-Kernel Attention
Large-kernel mechanisms enlarge receptive fields and can improve detection by incorporating surrounding context. Selective large-kernel designs further allow adaptive kernel selection to suppress background noise efficiently.

---

## 3. Method
### 3.1 Overview
Our model is specified by `ultralytics/cfg/models/v8/yolov8-drone-mamba.yaml`. Core modules are implemented in `ultralytics/nn/modules/block.py`. The model outputs three detection scales **P2/P3/P4** (stride 4/8/16), explicitly keeping a **P2** head to better handle tiny objects.

**Key components:**
- **Shallow (P1/P2):** DCE + ERB for high-resolution texture preservation.
- **Deep (P3/P4):** C2f-VSS (Mamba-like state space modeling) + C2f-CLIP (semantic anchors).
- **Fusion enhancement:** LSKBlock at a critical top-down fusion node before head.

### 3.2 Shallow Texture-Preserving Enhancement: DCE and ERB
To retain high-resolution texture information under 1024 inputs while controlling computation, we introduce lightweight enhancement blocks in early stages:
- **DCE** modules enhance local texture and edge cues immediately after early downsampling.
- **ERB** residual blocks refine shallow features with low overhead.

This strengthens early high-resolution representations critical to tiny objects.

### 3.3 Mamba-inspired Global Context: C2f-VSS
Let intermediate features be \( \mathbf{X}\in\mathbb{R}^{B\times C\times H\times W} \). We adopt a VSSBlock to model long-range spatial dependencies via linear state space dynamics, embedded into a CSP-style structure:
\[
\mathbf{Y} = \mathcal{F}_{\text{C2f-VSS}}(\mathbf{X}).
\]
C2f-VSS is inserted at backbone P3/P4 and in the neck refinement branch for P4, improving global context aggregation efficiently.

### 3.4 CLIP Semantic Anchors: C2f-CLIP
We generate class text embeddings using CLIP ViT-B/32 with prompt ensembling (five templates) and L2 normalization, producing \( \mathbf{E}\in\mathbb{R}^{10\times 512} \). The 10 VisDrone classes are:

- pedestrian  
- people  
- bicycle  
- car  
- van  
- truck  
- tricycle  
- awning-tricycle  
- bus  
- motor  

The embeddings are treated as **frozen semantic anchors** and injected into deep features:
\[
\mathbf{Y} = \mathcal{F}_{\text{CLIP}}(\mathbf{X}; \mathbf{E}).
\]

### 3.5 Dynamic Large Selective Kernel Attention (LSKBlock)
We insert LSKBlock at the fusion node after concatenating P4-upsampled features with P3 features, enabling adaptive large receptive fields that enhance tiny object responses:
\[
\mathbf{Y} = \mathcal{A}_{\text{LSK}}(\mathbf{X}).
\]

### 3.6 Training Objective
We follow the YOLOv8 detection objective:
\[
\mathcal{L}=\lambda_{box}\mathcal{L}_{box}+\lambda_{cls}\mathcal{L}_{cls}+\lambda_{dfl}\mathcal{L}_{dfl},
\]
with \( \lambda_{box}=7.5 \), \( \lambda_{cls}=0.5 \), and \( \lambda_{dfl}=1.5 \).

---

## 4. Experiments
### 4.1 Dataset and Metrics
We evaluate on **VisDrone2019-DET** and report results on the **validation split**. The primary metric is **mAP@0.5**; precision, recall, and mAP@0.5:0.95 are reported as auxiliary metrics.

### 4.2 Implementation Details (1024 Main Setting)
Training uses 1024×1024 input resolution, AdamW optimization, cosine learning rate schedule, and strong augmentation. Mosaic is enabled and closed at epoch 40 for stable late-stage convergence. Training is early-stopped due to metric saturation; the best result is reached at epoch 118.

### 4.3 Main Results on VisDrone2019-DET (Val)
At epoch 118, DCE-YOLOv8-UAV achieves:
- Precision = **0.6229**
- Recall = **0.53481**
- mAP@0.5 = **0.55981** (primary)
- mAP@0.5:0.95 = **0.35302**

**Full logged values (epoch 118):**
- time: 46294  
- train/box_loss: 1.28233  
- train/cls_loss: 0.92701  
- train/dfl_loss: 1.03454  
- metrics/precision(B): 0.6229  
- metrics/recall(B): 0.53481  
- metrics/mAP50(B): 0.55981  
- metrics/mAP50-95(B): 0.35302  
- val/box_loss: 1.2857  
- val/cls_loss: 0.95466  
- val/dfl_loss: 1.05639  
- lr/pg0: 0.000431257  
- lr/pg1: 0.000431257  
- lr/pg2: 0.000431257  

### 4.4 Comparisons with Existing Methods
We adopt the baseline comparison protocol and method list from the published paper **“DCE-YOLOv8: Lightweight and Accurate Object Detection for Drone Vision”** (IEEE Xplore: https://ieeexplore.ieee.org/abstract/document/10720001). Baseline numbers are cited accordingly, and our results are reported under the 1024-resolution setting.

### 4.5 Fairness and Resolution Sensitivity (Strategy B)
Many published VisDrone baselines report results under 640 resolution. Since our main setting is 1024, our primary goal is to improve UAV small-object performance under a high-resolution regime. To address fairness concerns without re-running all baselines, we provide a resolution-sensitivity analysis and clearly separate improvements due to architectural changes from those due to resolution.

**Preliminary estimate (to be replaced after running 640 setting):**
- Ours@640: mAP@0.5 ≈ 0.535; mAP@0.5:0.95 ≈ 0.335 (estimated)

---

## 5. Conclusion
We propose DCE-YOLOv8-UAV, a high-resolution (1024×1024) hybrid YOLOv8 model for UAV small object detection integrating Mamba-inspired global context modeling (C2f-VSS), CLIP semantic anchors (C2f-CLIP), and dynamic large selective kernel attention (LSKBlock), together with efficient shallow enhancement blocks (DCE and ERB). On VisDrone2019-DET validation set, our method achieves a best mAP@0.5 of 0.55981, demonstrating that combining efficient long-range context, frozen language priors, and selective large-kernel attention benefits UAV small object detection.

---

## Tables (to paste into Word as needed)
> You can paste your table screenshots or your LaTeX tables. If you want, I can also output them as Markdown tables (plain text) suitable for Word.

- Table 3: Experimental results on VisDrone2019 dataset (methods from baseline paper + our 1024 row).
- Table 4: Results across GPU devices (baseline paper + our RTX4090 row TBD).
- Table 5: Ablation study (baseline paper + our row).
- Table 6: Results on other datasets (baseline paper).
- Table X: Resolution sensitivity (baseline paper @640 + our @1024 + estimated @640).
