# Retinal_Segmentation
# Automated Retinal Layer Segmentation and Disease Biomarker Detection

## Overview

This repository contains the implementation of an advanced algorithm for automated segmentation of retinal layers and disease biomarkers, including regular drusen and reticular pseudodrusen (RPD), in spectral-domain optical coherence tomography (SD-OCT) images. The project combines graph-based shortest path algorithms with deep learning techniques to achieve subpixel accuracy in detecting and segmenting retinal layers and biomarkers associated with age-related macular degeneration (AMD).

---

## Features

- **Automated Segmentation:** Detects and segments 11 retinal layers using a hybrid approach combining deep learning-derived probability maps and graph-based shortest path methods.
- **Disease Biomarker Detection:** Separately identifies regular drusen and RPD, critical indicators of early AMD progression.
- **High Accuracy:** Achieves subpixel precision validated against manual expert annotations.
- **Scalability:** Designed for large-scale analysis of retinal morphology in clinical and research settings.

---

## Methodology

### 1. Dataset
- **Source:** SD-OCT images from 45 eyes (25 AMD-diagnosed, 20 healthy) .
- **Preprocessing:** Images were resized to standard dimensions for consistent analysis.

### 2. Algorithm Design
#### **Deep Learning Component**
- A U-Net architecture was used to generate probability maps for identifying layer boundaries and disease biomarkers.
- Training involved 831 B-scans from AMD-diagnosed eyes, with testing on 512 B-scans.

#### **Shortest Path Component**
- Dijkstra's algorithm was applied to compute optimal paths for layer segmentation based on edge weights derived from pixel intensity gradients and U-Net-generated probability maps.

#### **Hybrid Approach**
- Combines deep learning-derived probability maps with graph-based shortest path segmentation for robust detection across diverse retinal layers.

### 3. Validation
- Manual segmentation by certified OCT graders was used as ground truth to evaluate accuracy.
- The algorithm demonstrated mean differences within subpixel ranges, ensuring high reliability.

---

## Results

The algorithm successfully segmented:
- **11 Retinal Layers:** Including the inner limiting membrane (ILM), external limiting membrane (ELM), inner/outer retinal pigment epithelium (RPE), and choroid-sclera junction.
- **Disease Biomarkers:** Regular drusen and RPD were segmented separately, enabling detailed analysis of AMD progression.

Key metrics:
- Mean difference: $$-0.75 \pm 1.99$$ pixels for RPD segmentation.
- Absolute mean difference: $$1.53 \pm 1.47$$ pixels for RPD segmentation.

---

## Applications

This project has significant implications for:
- **Clinical Diagnosis:** Enhancing early detection of AMD features such as drusen and RPD.
- **Research:** Facilitating large-scale studies on retinal diseases like diabetic retinopathy and glaucoma.
- **Treatment Monitoring:** Tracking disease progression over time with automated tools.

---

## Technologies Used

- **Programming Language:** MATLAB
- **Hardware:** Intel i7 CPU, NVIDIA Quadro GPU
- **Deep Learning Framework:** U-Net architecture

---

## Installation & Usage Instructions

### Installation
1. Clone the repository.
