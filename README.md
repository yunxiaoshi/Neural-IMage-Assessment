## NIMA: Neural IMage Assessment

This is a PyTorch implementation of the paper [NIMA: Neural IMage Assessment](https://arxiv.org/abs/1709.05424) by Hossein Talebi and Peyman Milanfar. You can learn more from [this post at Google Research Blog](https://research.googleblog.com/2017/12/introducing-nima-neural-image-assessment.html).

## Implementation Details

+ The model was trained on the [AVA (Aesthetic Visual Analysis) dataset](http://refbase.cvc.uab.es/files/MMP2012a.pdf), which contains roughly 255,500 images. You can get it from [here](https://github.com/mtobeiyf/ava_downloader). I used 80% of the dataset for training, and 5,000 images for validation. **Note: there may be some corrupted images in the dataset, remove them first before you start training**.

+ I used a VGG16 pretrained on ImageNet as the base network of the model, for which I got a ~0.22 EMD loss on the 5,000 validation images. Haven't tried the other two options (MobileNet and Inception-v2) in the paper yet. # TODO

+ The learning rate setting differs from the original paper. I can't seem to get the model to converge with momentum SGD using an lr of 3e-7 for the conv base and 3e-6 for the dense block. Other settings are all directly mirrored from the paper.

+ The code now only supports python3.

## Usage

+ Set ```--train=True``` and run ```python main.py``` to start training. The average training time for one epoch with ```--batch_size=128``` is roughly 1 hour on a Titan Xp GPU. For evaluation, set ```--test=True``` instead.

+ I found [https://learning-rates.com/](https://learning-rates.com/) a very handy tool to monitor training in PyTorch in real time. You can check it out on how to use it. Remember do ```pip install lrs``` first if you are inclined to use it.

## Example Results

+ Here shows the predicted mean scores of some images from the AVA dataset. The ground truth is in the parenthesis.

![result1](http://7xrnzw.com1.z0.glb.clouddn.com/result7.jpg)

+ The predicted aesthetic ratings from training on the AVA dataset are sensitive to contrast adjustments. See below (bottom right is the original input).
![result2](http://7xrnzw.com1.z0.glb.clouddn.com/result_comp.jpg)

## Requirements

+ PyTorch 0.4.0
+ torchvision
+ numpy
+ Pillow
+ pandas (for reading the annotations csv file)
