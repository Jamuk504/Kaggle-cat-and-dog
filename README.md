# Kaggle Cat and Dog Image Classifier (PyTorch)

This project implements a deep learning model using PyTorch to classify images as either "cat" or "dog." The solution utilizes transfer learning based on the ResNet-18 architecture and features robust data cleaning and a custom training pipeline focused on achieving high accuracy through strategic fine-tuning.

##  Key Results

The model achieved highly competitive performance on the validation set by employing a targeted fine-tuning approach:

| Metric | Value |
| :--- | :--- |
| **Best Validation Loss** | **0.0507** |
| **Peak Accuracy** | **98.2%** |

##  Implementation & Features

The following key features and strategies were implemented to achieve stability and performance:

### 1. Strategic Fine-Tuning

The best results were achieved by strategically **unfreezing and fine-tuning only the final fully connected (FC) layer. This allowed the model to adapt the highest-level features to the specific texture and shape differences between cats and dogs, without disturbing the general, low-level feature extraction layers.

### 2. Comprehensive Data Cleaning

A critical step involved cleaning the raw dataset to ensure model reliability. The cleaning pipeline addressed three main issues:

| Issue | Detection Method |
| :--- | :--- |
| **Corrupt Files** | Checking if the image file could be successfully opened/loaded. |
| **Blank/Homogeneous Images** | Converting the image to RGB format and calculating the standard deviation (using NumPy). Images with a standard deviation below a set threshold were removed. |
| **System/Junk Files** | Checking file names for common system files (e.g., `Thumbs.db`) and removing them. |

### 3. Custom Training Pipeline

The training process was managed with a focus on efficiency and optimal model selection:

* **Optimizer & Scheduler:** The **Adam** optimizer was used in conjunction with a **ReduceLROnPlateau** scheduler to dynamically adjust the learning rate based on validation loss, helping the model escape local minima.
* **Early Stopping:** A cutoff system was devised where training automatically terminates if no further improvement in the validation loss is observed after a defined `patience` value, preventing overfitting and saving computational resources.
* **Model Checkpointing:** A state management system tracks the best validation loss (`vloss`) achieved so far, ensuring that **only models with superior performance** are saved to disk.
* **Logging:** Integration with **TensorBoard** was added to visualize training progress, loss curves, and other metrics in real-time.

### 4. Evaluation and Debugging

After training, a comprehensive evaluation system was built to calculate the final model accuracy. Crucially, the system also **prints samples of incorrectly identified images**, which provides valuable visual feedback for debugging the model and informing future optimization strategies.

##  Repository Structure and Setup

### Data Preparation

The initial data required pre-processing into four distinct folders to facilitate PyTorch's `ImageFolder` structure:

```
/dataset
├── training_set
│   ├── cat
│   └── dog
└── test_set
    ├── cat
    └── dog
```

### Configuration

All hyperparameters, file paths, and constant values are managed in a dedicated configuration file to ensure the codebase remains clean and easily modifiable.

##  Future Work

A subsequent attempt was made to further improve accuracy by unfreezing `layer4` of the ResNet-18 model in addition to the FC layer. It unfortunately resulted in a slightly worse performance (loss of 0.058) regardless of learning rate tweaks.

Future efforts will focus on:
1.  Exploring different learning rate schedules and weight decay settings for deeper fine-tuning.
2.  Implementing additional data augmentation techniques (e.g., Mixup or CutMix).
3.  Experimenting with more modern transfer learning architectures (e.g., Vision Transformers).
4.  Investigating the incorrectly classified images printed during evaluation to identify common failure modes.
