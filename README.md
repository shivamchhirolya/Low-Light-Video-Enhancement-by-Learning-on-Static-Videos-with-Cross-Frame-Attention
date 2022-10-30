# Low-Light-Video-Enhancement-by-Learning-on-Static-Videos-with-Cross-Frame-Attention

This is a pythorch implementation of “Low-Light-Video-Enhancement-by-Learning-on-Static-Videos-with-Cross-Frame-Attention” in BMVC 2022, by [Shivam Chhirolya](https://www.linkedin.com/in/shivam-chhirolya/), [Sameer Malik](https://www.linkedin.com/in/sameer-malik-20b45b153/), and [Rajiv Soundararajan](https://ece.iisc.ac.in/~rajivs/#/).

[Paper](https://arxiv.org/abs/2210.04290)

![alt text](https://github.com/shivamchhirolya/Low-Light-Video-Enhancement-by-Learning-on-Static-Videos-with-Cross-Frame-Attention/blob/main/Results/Visual%20Comparision.png)


# Requirements

pytorch 1.10.0\
cudatoolkit 10.2\
numpy 2.7\
opencv-python 4.5\
einops 0.4.1\
skimage


# Checkpoints
please download the checkpoints from below link and put those checkpoints in checkpoints directory

https://drive.google.com/drive/folders/1fdQVKbYXXpc4icS4GvuJJCY92T7ejmag?usp=sharing

# Usage

#### Testing ####
1. To test our model on DAVIS dataset run " test_DAVIS.py "
2. To test our model on DRV RGB dataset run " test_DRV.py "

#### Training ####
1. To train our model on DAVIS dataset run " train_DAVIS.py "
2. To train our model on DRV RGB dataset run " train_DRV.py "


# Model #

1. Davis_901.pth (This model is trained using synthetic low light Davis dataset)
2. DRV_901.pth   (This model is trained using RGB form of DRV dataset)






