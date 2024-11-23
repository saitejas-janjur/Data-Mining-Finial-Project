CIFAR-10 Model Evaluation
This repository contains Python scripts for evaluating three different deep learning models on the CIFAR-10 dataset:

VGG11 (vgg11_demo.py)
ResNet50 (resnet_demo.py)
Vision Transformer (ViT) (vision_tdemo.py)

Prerequisites
Ensure the following requirements are met before running the scripts:

Python
Python 3.7 or higher

Libraries
Install the required Python packages using pip:

pip install torch torchvision


Dataset
The CIFAR-10 dataset will be automatically downloaded when you run the scripts.

Pre-trained Models
Ensure the following .pth files are available in the respective directories:

../VGG/vgg.pth for the VGG model
../Vision Transformer/vit.pth for the Vision Transformer model
../Resnet/resnet.pth for the ResNet model

Note: Update the PATH variable in each script if the .pth files are located in a different directory.

How to Run
Navigate to the Demo folder and execute the following commands:

ResNet Demo
python resnet_demo.py

VGG Demo
python vgg11_demo.py

Vision Transformer Demo
python vision_tdemo.py

EDA Notebook
To explore the dataset and perform exploratory data analysis (EDA), open the Jupyter Notebook:

jupyter notebook CIFAR_10_EDA.ipynb


Outputs
Each script evaluates the respective model on the CIFAR-10 test dataset and provides:

Overall accuracy of the model
Class-wise accuracy for all 10 CIFAR-10 classes:
Plane
Car
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck


Example Output:

Accuracy of the network on the 10000 test images: 85.47%
Accuracy for class: plane  is 91.3%
Accuracy for class: car    is 88.4%
...


Notes
Make sure GPU support is enabled if available for faster evaluation. The scripts will automatically detect and use CUDA if it's installed.
If the .pth file paths are not valid, update the PATH variable in each script to point to the correct directory.
