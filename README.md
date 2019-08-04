## NIMA: Neural IMage Assessment

This is a PyTorch implementation of the paper [NIMA: Neural IMage Assessment](https://arxiv.org/abs/1709.05424) by Hossein Talebi and Peyman Milanfar. You can learn more from [this post at Google Research Blog](https://research.googleblog.com/2017/12/introducing-nima-neural-image-assessment.html).

## Implementation Details

+ The model was trained on the [AVA (Aesthetic Visual Analysis) dataset](http://refbase.cvc.uab.es/files/MMP2012a.pdf), which contains roughly 255,500 images. You can get it from [here](https://github.com/mtobeiyf/ava_downloader). **Note: there may be some corrupted images in the dataset, remove them first before you start training**.

+ I split the dataset into 229,981 images for training, 12,691 images for validation and 12,818 images for testing.

+ I used a VGG16 pretrained on ImageNet as the base network of the model, for which I got a ~0.075 EMD loss on the 12,691 validation images. Haven't tried the other two options (MobileNet and Inception-v2) in the paper yet. # TODO

+ The learning rate setting differs from the original paper. I can't seem to get the model to converge with momentum SGD using an lr of 3e-7 for the conv base and 3e-6 for the dense block. Also I didn't do much hyper-param tuning therefore you could probably get better results. Other settings are all directly mirrored from the paper.

+ The code now only supports python3.

## Usage

+ Set ```--train=True``` and run ```python main.py``` to start training. The average training time for one epoch with ```--batch_size=128``` is roughly 1 hour on a Titan Xp GPU. For evaluation, refer to ```test.py``` for usage.

+ I found [https://lera.ai/](https://lera.ai/) a very handy tool to monitor training in PyTorch in real time. You can check it out on how to use it. Remember do ```pip install lera``` first if you are inclined to use it.

## Training Statistics

Training is done with early stopping monitoring. Here I set ```patience=5```.
![loss](https://i.ibb.co/p3srn3D/loss.png)

## Pretrained Model
[Google Drive](https://drive.google.com/file/d/1zRU3HhQDyv6KEPK1zBJS0vAWbNo7xRxS/view?usp=sharing)

## Annotation CSV Files
[Train](https://drive.google.com/file/d/1w313GtuqEBp0qqavdKSYHst-AAbQSTmq/view?usp=sharing) [Validation](https://drive.google.com/file/d/1GsrkIdn7Jcg--2y3iuuDqvEpc_6oV36w/view?usp=sharing) [Test](https://drive.google.com/file/d/17yvYHOc3CMq-04ZDri7BieXqwh2H633c/view?usp=sharing)

## Example Results

+ Here shows the predicted mean scores of some images from the validation set. The ground truth is in the parenthesis.

<p align="center">
<img src="https://i.ibb.co/8zqsss9/excellent-min.png">
</p>

+ Also some failure cases...

<p align="center">
<img src="https://i.ibb.co/x5x18B8/horrible-min.png">
</p>

+ The predicted aesthetic ratings from training on the AVA dataset are sensitive to contrast adjustments. Below images from left to right in a row-major order are with progressively sharper contrast. Upper rightmost is the original input.
<p align="center">
<img src="https://i.ibb.co/QvtrvBV/compare-min.png">
</p>

## Requirements

+ PyTorch 0.4.0+
+ torchvision
+ numpy
+ Pillow
+ pandas (for reading the annotations csv file)
