# Image Segmentation on COCO-2017 Dataset

## 7PAM2015-0509-2024 -- Research Methods in Data Science
### Mohit Agarwal (Student ID-22031257)

## Project Overview
This project implements instance segmentation on the COCO-2017 dataset, focusing specifically on four classes: cake, car, person, and dog. The implementation uses Mask R-CNN with a ResNet-50 backbone and Feature Pyramid Network (FPN) for accurate object detection and instance segmentation.

## Dataset
The project uses a subset of the COCO-2017 dataset:
- Training set: 300 images with annotations
- Validation set: 300 images with annotations
- Test set: 30 images

The dataset is organized in COCO format with JSON annotation files containing information about object instances, categories, and segmentation masks.

## Model Architecture
- **Base Model**: Mask R-CNN
- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **Target Classes**: cake, car, person, dog

## Project Structure
```
Assignment1/
├── coco_model_train.ipynb      # Main training and evaluation notebook
├── README.md                   # This file
├── RM_Segmentation_Assignment_dataset/
│   ├── train-300/              # Training dataset
│   │   ├── data/               # Training images
│   │   └── labels.json         # Training annotations
│   ├── validation-300/         # Validation dataset
│   │   ├── data/               # Validation images
│   │   └── labels.json         # Validation annotations
│   └── test-30/                # Test dataset
├── Images/                     # Directory for saving output images
└── MaskRCNN_Checkpoints/       # Directory for saving model checkpoints
```

## Configuration
The model is trained with the following configuration:
- Batch size: 5
- Number of epochs: 10
- Learning rate: 0.0001
- LR scheduler step size: 3
- LR scheduler gamma: 0.1
- Device: CUDA (if available) or CPU
- Random seed: 42

## Implementation Details
The implementation includes:
1. Custom COCO dataset loader to filter target classes
2. Mask R-CNN model with pre-trained weights
3. Training pipeline with optimizer and learning rate scheduler
4. Evaluation metrics including mean Average Precision (mAP)
5. Visualization tools for displaying predictions

## Requirements
- ipykernel==6.29.5
- ipython==9.3.0
- jupyter_client==8.6.3
- jupyter_core==5.8.1
- matplotlib==3.10.3
- matplotlib-inline==0.1.7
- opencv-python==4.11.0.86
- pandas==2.3.0
- pycocotools==2.0.10
- seaborn==0.13.2
- torch==2.7.1
- torchaudio==2.7.1
- torchvision==0.22.1
- tqdm==4.67.1

## Usage
1. Clone the repository
2. Set up the environment with required dependencies
3. Run the `coco_model_train.ipynb` notebook to train and evaluate the model

## Results
The model is evaluated on the validation set using standard COCO metrics:
- mAP (mean Average Precision)
- AP at different IoU thresholds
- AP for different object sizes (small, medium, large)

Visualizations of the segmentation results are saved in the Images directory.

## License
This project is part of the Research Methods in Data Science course assignment.

## Acknowledgments
- COCO dataset creators
- PyTorch and torchvision teams
- Research Methods in Data Science course instructors