# HPDDRNet
 A High-Precision Dual-Branch Deep Residual Network for Enhanced SEM Image Segmentation


# Abstract
SEM image segmentation is crucial for analyzing microscopic details in various scientific and industrial applications. However, existing dual-branch algorithms face challenges in effectively capturing both local details and global context when applied to SEM images. We present HPDDRNet, a precision-enhanced dual focusing deep residual network designed for advanced SEM image segmentation. HPDDRNet introduces a novel dual-branch architecture, integrating an ECA-enhanced high-resolution branch (ECAHRB) and a PAM-enhanced low-resolution branch (PAMLRB). The ECAHRB employs an efficient channel attention mechanism to enhance feature extraction and processing, ensuring detailed preservation in high-resolution features. Simultaneously, the PAMLRB leverages positional attention to capture long-range dependencies in low-resolution features, significantly improving the model’s global performance. We also propose a bilateral fusion strategy for effectively merging multi-scale features from both branches. Our extensive experiments demonstrate that HPDDRNet consistently outperforms existing state-of-the-art models, achieving higher segmentation accuracy and robustness across various complex SEM image scenarios. This approach demonstrates the effectiveness of our dual-branch channel attention mechanism in handling intricate visual tasks in SEM imaging.

# Introduction
This project is based on the MMSegmentation framework and implements a simple and efficient dual-branch focusing residual network for SEM image segmentation. Prior to running the code, you need to install the dependencies required by MMSegmentation. For detailed instructions, refer to [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).
Project Directory Structure:
- `configs/`： Contains configuration files for the project, used to set model parameters, training, and testing options.
    - `hpddrnet/`: Folder for HPDDRNet model configurations
        - `hpddrnet_23-slim_in1k-pre_2xb6-120k_rocks-1024x1024.py`: Configuration file for the HPDDRNet model
    - `...`: Other configuration files
- `mmseg/`：Contains the segmentation model code, likely based on MMSegmentation for SEM image segmentation tasks.
    - `models/`: Model implementation files
        - `backbones/`: Backbone network files
            - `hpddrnet.py`: Backbone implementation file for HPDDRNet
        - `...`: Other model files
    - `...`: Other directories or files related to MMSegmentation
       
- `tools/`：Utility scripts used in the project, including data preprocessing, post-processing, and auxiliary analysis.
- `README.md`：Project documentation file, explaining directory structure, installation instructions, and usage guidelines.
- `requirements.txt`：List of Python dependencies required for the project.
- `test.py`：Script for evaluating model performance, allowing for different configurations or parameters.
- `train.py`：Script for training the model with specific configurations or custom parameters.


# Datasets
The dataset can be downloaded from either [Google Driver](https://drive.google.com/file/d/1sNiqTctge2mS8GGcXRYOnqSoCOK_ZavJ/view?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/1YQI3O_aWJmCqR_rQ17qv-Q?pwd=1234)(password: 1234)进行下载


# Installation
1. Clone the repository:
``` bash
git clone https://github.com/fanjuchen/HPDDRNet.git
``` 
2. Install dependencies:
``` bash
pip install -r requirements.txt
``` 

# Train
Use the following command to start training the model:
``` python
python train.py configs/hpddrnet/hpddrnet_23-slim_in1k-pre_2xb6-120k_rocks-1024x1024.py
```
This command utilizes the hpddrnet_23-slim_in1k-pre_2xb6-120k_rocks-1024x1024.py configuration file to train the model with specified parameters.
# Testing
Use the following command to evaluate the trained model:
``` python
python test.py configs/hpddrnet/hpddrnet_23-slim_in1k-pre_2xb6-120k_rocks-1024x1024.py
```
This command uses the configuration file to test the model’s performance on the test set.