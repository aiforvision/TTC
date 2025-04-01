# A Tale of Two Classes: Adapting Supervised Contrastive Learning to Binary Imbalanced Datasets

[![Paper](https://img.shields.io/badge/paper-arxiv.2503.17024-red)](https://arxiv.org/abs/2503.17024) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

Official PyTorch implementation for the paper: **A Tale of Two Classes: Adapting Supervised Contrastive Learning to Binary Imbalanced Datasets**.

David Mildenberger*, Paul Hager*, Daniel Rueckert, Martin J. Menten (* equal contribution)

Technical University of Munich, Munich Center for Machine Learning, Imperial College London

---

**Abstract:** Supervised contrastive learning (SupCon) has proven to be a powerful alternative to the standard cross-entropy loss for classification of multi-class balanced datasets. However, it struggles to learn well-conditioned representations of datasets with long-tailed class distributions. This problem is potentially exacerbated for binary imbalanced distributions, which are commonly encountered during many real-world problems such as medical diagnosis. In experiments on seven binary datasets of natural and medical images, we show that the performance of SupCon decreases with increasing class imbalance. To substantiate these findings, we introduce two novel metrics that evaluate the quality of the learned representation space. By measuring the class distribution in local neighborhoods, we are able to uncover structural deficiencies of the representation space that classical metrics cannot detect. Informed by these insights, we propose two new supervised contrastive learning strategies tailored to binary imbalanced datasets that improve the structure of the representation space and increase downstream classification accuracy over standard SupCon by up to 35%.

![Concept Figure](https://github.com/aiforvision/TTC/blob/main/assets/fig1_concept.png)
*Standard SupCon collapses embeddings for binary imbalanced data, while our proposed methods maintain class separation.*

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/aiforvision/TTC.git
    cd TTC
    ```

2.  **Create a conda environment (Recommended):**
    ```bash
    conda create -n ttc python=3.11 
    conda activate ttc
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Datasets

We use the following datasets in our experiments:

1.  **iNaturalist 2021 (iNat21) Subsets:**
    *   We create binary subsets from iNat21 for:
        *   Plants (Oaks vs. Flowering Plants)
        *   Insects (Bees vs. Wasps)
        *   Mammals (Hoved Animals vs. Carnivores)
    *   Artificial imbalance is introduced by subsampling the majority class to achieve 99%-1%, 95%-5%, and 50%-50% ratios, keeping the total sample count constant within each subset category.
    *   Download iNat21 from the [official source](https://github.com/visipedia/inat_comp/tree/master/2021). Configs for the specific subsets used can be found in `configs/data` .

2.  **Medical Datasets:**
    *   **UK Biobank Cardiac MRI (Infarction):** Curated subset. Due to data privacy, this dataset requires application via the [UK Biobank](https://www.ukbiobank.ac.uk/). Preprocessing details are in the paper's supplement / data scripts. Imbalance: ~4%.
    *   **MedMNIST:**
        *   `BreastMNIST`: Imbalance ~37%.
        *   `PneumoniaMNIST`: Imbalance ~35%.
        *   Download via the [official MedMNIST repository](https://medmnist.com/) or use the provided data loaders.
    *   **FracAtlas:** Fractures dataset. Imbalance ~21%. Download from the [official source](https://github.com/M3DV/FracAtlas).

**Data Preparation:** Update the paths in the configuration files. Specific preprocessing steps might be required as detailed in `data/`.

---

## Usage

This project uses [Hydra](https://hydra.cc/) for configuration management. Experiments are launched using `train.py`, and behaviour is controlled via command-line overrides or by modifying the YAML configuration files located in the `configs/` directory.

### 1. Contrastive Pre-training

Run contrastive pre-training experiments using the following command structure. Specify the method and dataset details via Hydra overrides.

**Base Command:**

```bash
python train.py experiment=<experiment_type> experiment/specs=<dataset_spec> [options...]
```



### 1. Pre-training (Contrastive Learning)

**Example Commands:**

*   **Standard SupCon (Baseline):**
    ```bash
    python train.py \
        experiment=contrastive \
        # ... other args ...
    ```

*   **Supervised Minority (Ours):** Set `ratio_supervised_majority` to `0.0`.
    ```bash
    python train.py \
        experiment=contrastive \
        module.ratio_supervised_majority=0.0 \
        # ... other args ...
    ```

*   **Supervised Prototypes (Ours):**
    ```bash
    python train.py \
        experiment=contrastive_sup_prototype \
        # ... other args ...
 
    ```

*   **Weighted Cross-Entropy (Baseline):**
    ```bash
    python train.py \
        experiment=weighted_ce \
        # ... other args ...
    ```

**Key Arguments:**

*   `experiment`: Selects the main training configuration (`contrastive`, `contrastive_sup_prototype`, `weighted_ce`). Defaults defined in `configs/experiment/`.
*   `experiment/specs`: Selects dataset-specific configurations (e.g., `plants`, `animals`, `insects`,`infarction`,). Defaults defined in `configs/experiment/specs/`.
*   `class_ratios`: List defining the ratio for each class (e.g., `[0.01, 0.99]` for 1% minority). Crucial for artificial imbalance datasets.
*   `module.ratio_supervised_majority`: Float between 0.0 and 1.0. Controls the degree of supervision applied to the majority class in the `contrastive` experiment. `0.0` corresponds to **SupMin** (NT-Xent on majority), `1.0` corresponds to standard SupCon.
*   `batch_size`: Batch size.
*   `module.*`: Parameters related to the model, optimizer, loss function, learning rate schedule (e.g., `module.lr`, `module.optimizer_name`, `module.warmup_epochs`). See `configs/module/` or the relevant experiment config.
*   `trainer.*`: Parameters for the PyTorch Lightning Trainer (e.g., `trainer.max_epochs`, `trainer.devices`). See `configs/trainer/`.
*   `data.*`: Data loading and processing parameters. See `configs/data/`.



**Other Baselines (SBC, BCL, PaCo, KCL, TCL):**

Implementations for SBC, BCL, and PaCo are available in their official repositories:

*   **SBC:** [https://github.com/JackHck/SBCL](https://github.com/JackHck/SBCL)
*   **PaCo:** [https://github.com/dvlab-research/Parametric-Contrastive-Learning](https://github.com/dvlab-research/Parametric-Contrastive-Learning)
*   **BCL:** [https://github.com/FlamieZhu/Balanced-Contrastive-Learning](https://github.com/FlamieZhu/Balanced-Contrastive-Learning)
*   **KCL:** see `loss.py`
*   **TSC:** see `loss.py`

Please refer to these repositories for instructions on running these methods.


## Fine-tuning

To fine-tune a pre-trained encoder on a downstream task set experiment to finetune.



### Fine-tuning Options

- `experiment`: Type of fine-tuning (`finetune`, `finetune_prototype`)
- `base_model_path`: Path to the pre-trained encoder checkpoint
- `data.balanced_batches`: Whether to balance batches during training
- `data.data_module.subsample_balanced`: Downsample both classes to match the minority class size
    For finue tuning on a subset of the train set, set this flag and class ratio to the class size.

## Metrics

Find the used metrics in `/metrics/metrics.py`.

---

## Citation

If you find this code useful for your research, please cite our paper:

```
@inproceedings{mildenberger2025tale,
  title={A Tale of Two Classes: Adapting Supervised Contrastive Learning to Binary Imbalanced Datasets},
  author={Mildenberger, David and Hager, Paul and Rueckert, Daniel and Menten, Martin J.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

## Acknowledgments

We based some of our code on [SupCon](https://github.com/HobbitLong/SupContrast) and thank the authors for making their code publicly available.

