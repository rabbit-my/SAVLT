# SAVLT:Structure-Aware Vision-Language Tuning for Multi-Center Cervical OCT Diagnosis

⭐ Star this repo if you find it useful!

## 📝 Abstract

This study proposes a novel method for adapting CLIP to **cervical OCT** diagnosis by leveraging textual consistency to address **cross-center image variations**. Our approach utilizes LoRA to fine-tune CLIP efficiently and incorporates a dual-alignment mechanism to enhance consistency between image and text features across centers. Experimental results on **three datasets** demonstrate that our method outperforms existing SOTA approaches in various metrics and cross-center generalization performance, showcasing the potential of combining cross-modal learning with OCT imaging to advance automated cervical disease detection. 
<p><strong>Tags:</strong> <code>Medical Imaging</code> <code>Cervical OCT </code> <code>CLIP Fine-tuning</code></p>


## ⚡️ Quick Start



### 📦 Installation

The following instructions are for Linux installation. We would like to recommend the **requirements** as follows.

```
conda create -n your_env python==3.9
conda activate your_env
pip install -r requirements.txt

# install CLIP
git clone https://github.com/openai/CLIP.git
cd CLIP
pip install .
```
This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the dassl environment. 

```
git clone https://github.com/KaiyangZhou/Dassl.pytorch
cd Dassl.pytorch-master
pip install .
```

### 📁 Dataset Prepare

We follow the dataset from [“Prior Activation Map Guided Cervical OCT Image Classification”](https://link.springer.com/chapter/10.1007/978-3-031-72384-1_36) (MICCAI, 2024), which is collected based on the study [“Multi-center clinical study using optical coherence tomography for evaluation of cervical lesions in-vivo”](https://www.nature.com/articles/s41598-021-86711-3)(Scientific Reports, 2021).

Please create a folder named DATA and place the dataset inside it, organized in the following structure:

```
$DATA/
|–– mixed/
    |–– images
        |–– class1
            |–– ×××1.png
            |–– ×××2.png
            ...
        |–– class2
        ...
        |–– class5
|–– huaxi/
    ...
|–– xiangya/
    ...
```
You can replace **mixed**, **huaxi**, and **xiangya** with your own dataset.

**Load dataset** in `datasets/.`According to the paper, we select data from the mixed-center as the few-shot support set.
```
datasets/
|–– __init__.py
|–– huaxi_oct.py # Return the test set(External B dataset).
|–– mixed_center_oct.py  # Return the training set, validation set, and test set.
|–– xiangya.py # Return the test set (External A dataset).
|–– utils.py
```

Our code supports two options for dataset splits: using separate JSON files for training, validation, and testing, or performing random splits based on a given ratio. See `datasets/mixed_center_oct.py` for details.

### 🚀 Training & Evaluation

We written a runnable script using 32-shot as an example, with parameters that can be easily modified as needed：
```
python run_shell.py
```
You can set the result saving location in the `run_shell.py` file by modifying `output_dir = ""`.



## 💬 Contact

For questions or collaborations:


👉 [Submit an Issue](https://github.com/rabbit-my/SAVLT/issues)

Thank you for your attention!

