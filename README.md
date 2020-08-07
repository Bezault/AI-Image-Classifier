# Intro

You have a lot of images from a test campaign and you want to find bugs?  
AI Image Classifier allows to classify images into a given number of clusters.  
Thus organizing similar images together, you'll be able to find issues more quickly.  

# Tutorial

## Prerequisites
1. Download [ResNet50 weights without top layer](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5) into code directory
2. All images have to be in the same directory

## Procedure
1. Click on File/Open
2. Select one image into your image directory
3. Setup Cluster numbers (you can start with 5 if you have no idea)
4. Click "Analyze"
5. Look into console the Extract Features progression % and clustering
6. Clusters will be displayed on the right
7. Click one image into cluster and click show to see what it is
8. To save current results : Click "Save"
9. To change cluster value : change cluster number and click "Update"
10. To load previous results : Click "Load"
11. "Preview" function is still in development, if you want to help you are kindly invited to update code. This function aims to show a matrix with images of the same cluster. This will allow to see quickly what's inside the cluster.
