# AI-Powered Detection and Localization of Brain Tumors Using MRI Scans

This repository contains a deep learning-based solution for detecting and localizing brain tumors using MRI scans. Utilizing a layered pipeline of ResNet and ResUNet models, the project provides an efficient and accurate method for medical image classification and segmentation. Transfer learning is also used to save computational time and costs.

## Project Overview:

Brain tumor detection and localization play a crucial role in early diagnosis and treatment planning. This project leverages the power of Convolutional Neural Networks (CNNs) to classify MRI images and pinpoint tumor locations at a pixel level, offering a robust solution for healthcare professionals.

## Key Components:

__ResNet Model__: Classifies brain MRI scans to detect the presence of tumors.

__ResUNet Model__: Segments and localizes tumors in detected cases, providing pixel-level accuracy.

__Transfer Learning__: Utilizes a pre-trained ResNet50 model on the ImageNet dataset to accelerate training and reduce computational requirements.

__Streamlined Data Handling__: Processes large MRI datasets, including normalization, cross-validation splits, and batch training.

## Project Structure:

### Files and Folders:

Brain_Tumor.ipynb:
The main Jupyter Notebook that contains the entire pipeline, including data preprocessing, model training, evaluation, and visualization of results.

data.csv:
Metadata file containing the image paths and labels for the MRI scans.

data_mask.csv:
Metadata file mapping each MRI scan to its corresponding mask, indicating the location of the tumor for segmentation.

resnet-50-MRI.json:
JSON file defining the architecture of the ResNet50 model used for classification.

ResUNet-MRI.json:
JSON file defining the architecture of the ResUNet model used for segmentation and tumor localization.

weights.hdf5:
Pre-trained weights for the ResNet50 model, saved in HDF5 format.

weights_seg.hdf5:
Pre-trained weights for the ResUNet model, saved in HDF5 format.

utilities.py:
A Python script containing helper functions used across the project, such as data preprocessing utilities and image augmentation techniques.

## Dataset:

The dataset consists of 3,929 MRI images. Each image has an associated mask, which identifies regions containing tumors. This project uses data.csv and data_mask.csv to organize and process the images for training and evaluation. Key attributes in these files:

__Image Path__: Location of each MRI scan.

__Mask Path__: Location of the corresponding mask image, highlighting the tumor area.

__Patient ID__: Unique identifier for each patient record in the dataset.

The MRI images are preprocessed by resizing, normalization, and cross-validation split creation to ensure efficient training and testing. The data has been structured for use in both classification and segmentation tasks.

## Accessing the Dataset and Models:

Due to the large size of the dataset and pre-trained model files, these assets are stored in Google Drive. You can access them via https://drive.google.com/drive/folders/1CsN9RNYHVhi5JfU-lKC4p2oevZHswhof. The folder contains:

data.csv and data_mask.csv for managing image and mask paths.

weights.hdf5 and weights_seg.hdf5 for the pre-trained ResNet50 and ResUNet models, respectively.

Additional files like resnet-50-MRI.json and ResUNet-MRI.json to define model architectures and also the main code file Brain_Tumor.ipynb. It also contains a code file named utilities.py.

## Model Overview

__ResNet50__ - Tumor Classification

The ResNet50 model, a widely-used deep residual network, is employed here for binary classification:

Input: Preprocessed MRI images.

Output: Binary classification indicating the presence or absence of a tumor.

Transfer Learning: Leveraging a pre-trained ResNet50 from the ImageNet dataset, this model achieves high accuracy with minimal training time.

__ResUNet__ - Tumor Localization and Segmentation

The ResUNet model combines the ResNet architecture with the U-Net segmentation model, facilitating pixel-wise tumor localization:

Input: MRI images with detected tumors.

Output: Segmentation mask highlighting the tumor region.

Pixel-Level Accuracy: Allows for precise localization, which is crucial for medical imaging tasks.

__Transfer Learning__

Transfer learning is applied to both models:

Accelerated Training: Starting with pre-trained weights on a large-scale dataset like ImageNet enhances the models’ learning capability with limited training data.

Computational Efficiency: Reduces the need for extensive resources by leveraging knowledge from the initial training phase.

## Model Performance:

Both models were evaluated using key metrics such as accuracy, precision, recall, and F1 score:

__Classification Accuracy__: The ResNet50 model achieved an accuracy of 98%, ensuring reliable classification results.

__Segmentation Precision__: The ResUNet model demonstrated excellent performance in localizing tumors, with high recall and F1 scores.

## Usage Instructions

__Prerequisites__:

Python 3.x

GPU

TensorFlow, Keras, OpenCV, and other required libraries

## Running the Models:

Clone the Repository: Download all files from this repository.

Download Dataset and Weights: Access the dataset and model weights from the Google Drive link and place them in the appropriate directories.

Run Brain_Tumor.ipynb: This notebook will walk you through the steps from data loading and preprocessing to model training and evaluation.

## Future Enhancements

__Hyperparameter Tuning__: Further optimize model parameters to improve performance.

__Additional Data__: Integrate larger datasets for enhanced model robustness.

__Real-Time Deployment__: Develop a live web app using Streamlit or Flask for real-time tumor detection and localization.
