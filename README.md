# Vesuvius Challenge: First Letters Prize (Scrolls 2, 3, & 4)

## 1. Introduction
This repository documents my findings and experimental architectures developed while pursuing the **First Letters Prizes** for Scrolls 2, 3, and 4. 

The core hypothesis of this research was that ink should be captured by a model regardless of scroll texture. By incorporating Scroll 1 into the training data, I aimed to generalize ink detection. However, the process revealed that ink detection is significantly more complex than identifying simple surface cracks; data bias often leads models to capture very specific qualities of ink.

### Key Observation
Regardless of the specific architecture, most models were capable of learning "something" from the data, showing surprisingly consistent performance on Scroll 1 across various configurations.

---

## 2. Model Architectures
I experimented with a wide variety of architectures. Below are the primary models tested (excluding custom variants and failed experiments):

* **TransformerUNet** (Current Best)
* **TimeSformer** (From Grand Prize)
* **ResNet + Decoder** (From Grand Prize)
* **RCAN** (Residual Channel Attention Networks)
* **HSDT** (Hierarchical Swin Discrete Transformer)
* **UNet_CT_Multi_Att_DSV_3D**
* **MIRNet**
* **QRNN3D** (Quasi-Recurrent Neural Networks)
* **ConvLSTM U-Net**
* **VideoFocalNet**

### Custom Variants
Beyond the base models, I "mixed and matched" components from multiple papers, tweaking:
* **Loss & Activation:** Custom functions to handle ink sparsity.
* **Layers:** Variations in pooling, number of layers, and attention mechanisms.
* **Optimizers:** Fine-tuning learning rates and scheduling.
* **Hybridization:** Running models in parallel or sequence to combine feature extraction strengths.

---

## 3. Methodology

### The "Chemistry" Approach
My training philosophy mirrored **research chemistry**:
1.  **Small-Scale Testing:** Using limited segments of each scroll on a **T4 GPU** to test hypotheses.
2.  **Production Scaling:** Once a "reaction" (architecture) proved successful, I scaled up to an **L4 GPU**, pouring in maximum training data to push the idea to its limits.

### Experiment Tracking & Infrastructure
* **Environment:** Google Colab (T4/L4 GPUs).
* **Storage:** Segments managed via Google Drive; local storage for 20GB+ of prediction outputs.
* **Logging:** Tracked via Weights & Biases (**wandb**).
    * **2,300+** total training hours logged.
    * **1,815** models trained.

---
