# GradCAM Implementation on Shenzhen-Hospital-CXR-Set

Gradient-weighted Class Activation Mapping(Grad-CAM), uses the gradients of any target concept (say ‘dog’ in a classification network or a sequence of words in captioning network) flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.

**Implementing Grad-CAM on Shenzhen-Hospital-CXR-Set**
for the implementation of the Grad-CAM we require a CNN model for which I am using DensNet121 which is a pretrained model trained on multiple chestXray datasets.
# DenseNet 224x224 model trained on multiple datasets
model = xrv.models.DenseNet(weights="densenet121-res224-all")
