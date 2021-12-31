# Object-Shadow-Generation-Dataset-DESOBA


**Object Shadow Generation** is to deal with the shadow inconsistency between the foreground object and the background in a composite image, that is, generating shadow for the foreground object according to background information, to make the composite image more realistic.

Our dataset **DESOBA** is a synthesized dataset for Object Shadow Generation. We build our dataset on the basis of Shadow-OBject Association dataset [SOBA](https://github.com/stevewongv/InstanceShadowDetection),  which  collects  real-world  images  in  complex  scenes  and  provides annotated masks for object-shadow pairs.  Based on SOBA dataset, we remove all the shadows to construct our DEshadowed Shadow-OBject Association(DESOBA) dataset, which can be used for shadow generation task and other shadow-related tasks as well. We illustrate the process of our DESOBA dataset construction based on SOBA dataset in the figure below.




<img src='Examples/task_intro.png' align="center" width=1024>

Illustration of DESOBA dataset construction: The green arrows illustrate the process of acquiring paired data for training and evaluation.  Given a ground-truth target image I<sub>g</sub>, we manually remove all shadows to produce a deshadowed image I<sub>d</sub>. Then, we randomly select a foreground object in I<sub>g</sub>, and replace its shadow area with the counterpart in I<sub>d</sub> to synthesize a composite image I<sub>c</sub> without foreground shadow. I<sub>c</sub> and I<sub>g</sub> form a pair of input composite image and ground-truth target image. 
The red arrow illustrates our shadow generation task. Given I<sub>c</sub> and its foreground mask M<sub>fo</sub>, we aim to generate the target image I<sub>g</sub> with foreground shadow.


 Our DESOBA dataset contains 840 training images with totally 2,999 object-shadow pairs and 160 test images with totally 624 object-shadow pairs. The DESOBA dataset is provided in [**Baidu Cloud**](https://pan.baidu.com/s/1fYqcSjGSr52jppg2LEA1qQ) (access code: sipx), or [**Google Drive**](https://drive.google.com/file/d/114BU47G0OJV3vmx5WKxGnWDSj2Bzh6qS/view?usp=sharing).
 
 <img src='Examples/dataset-samples.png' align="center" width=1024>

|DESOBA Split|Num of Images|Num of Pairs| 
|:--:|:--:|:--:|
|Train Set| 839| 11509 |
|Test set| 160 |160|915|


## Visualizing training/testing pairs of DESOBA dataset

`cd ./DESOBA_DATASET_util`

- visualizing train pairs (839 train images with 11509 pairs):
`python Vis_Desoba_Dataset --serial_batches --isTrain 1`,
train pairs are store in /DESOBA_DATASET/TrainTestVisualization/train/

- visulizing test bosfree pairs (34 test bosfree images with 34 pairs):
`python Vis_Desoba_Dataset --serial_batches --isTrain 0 --bosfree`,
test bosfree pairs are store in /DESOBA_DATASET/TrainTestVisualization/train/test_bosfree

- visulizing test bos pairs (126 test bos images with 581 pairs):
`python Vis_Desoba_Dataset --serial_batches --isTrain 0 --bos`,
test bos pairs are store in /DESOBA_DATASET/TrainTestVisualization/train/test_bos

We show some examples of training/testing tuples in below:
<img src='/data_processing/Visualization_Examples/9.png' align="center" width=1024>
<img src='/data_processing/Visualization_Examples/5.png' align="center" width=1024>
<img src='/data_processing/Visualization_Examples/6.png' align="center" width=1024>
<img src='/data_processing/Visualization_Examples/12.png' align="center" width=1024>
from left to right: synthetic composite image without foreground shadow, target image with foreground shadow, foreground object mask, foreground shadow mask, background object mask, and background shadow mask.


## Producing real composite images from test images of DESOBA

`cd ./DESOBA_DATASET_util`

- producing real composite images with one foreground object:
`python Vis_RealCompositionImages_fromDesoba_Dataset.py --foreground_object_num 1`, real composite images with one foreground object are store in /DESOBA_DATASET/CompositeImages/1_ForegroundObject/

- producing real composite images with two foreground objects: `python Vis_RealCompositionImages_fromDesoba_Dataset.py --foreground_object_num 2`,
real composite images with one foreground object are store in /DESOBA_DATASET/CompositeImages/2_ForegroundObject

We show some examples of real composite images in below:
<img src='/data_processing/Visualization_Examples/r1.png' align="center" width=1024>
<img src='/data_processing/Visualization_Examples/r2.png' align="center" width=1024>
<img src='/data_processing/Visualization_Examples/r3.png' align="center" width=1024>
<img src='/data_processing/Visualization_Examples/r4.png' align="center" width=1024>
from left to right: synthetic composite image without foreground shadow,  foreground object mask

## Dataloader preparation for your own project
We provide the code of obtaining training/testing tuples, each tuple contains foreground object mask, foreground shadow mask, background object mask, background shadow mask, shadow image, and synthetic composite image without foreground shadow mask. The dataloader is available in `/DESOBA_DATASET_util/data/DesobaSyntheticImageGeneration_dataset.py`, which can be used as dataloader in training phase or testing phase.


# Our SGRNet

Here we provide PyTorch implementation and the trained model of our SGRNet.

## Prerequisites

- Python 
- Pytorch
- PIL

## Getting Started

## Installation

- Clone this repo:

```bash
git clone https://github.com/bcmi/Object-Shadow-Generation-Dataset-DESOBA.git
cd Object-Shadow-Generation-Dataset-DESOBA
```

- Download the DESOBA dataset from  [**Baidu Cloud**](https://pan.baidu.com/s/1fYqcSjGSr52jppg2LEA1qQ) (access code: sipx), or [**Google Drive**](https://drive.google.com/file/d/114BU47G0OJV3vmx5WKxGnWDSj2Bzh6qS/view?usp=sharing).


- Download the pretrained model of SGRNet and baseline methods from [**Baidu Cloud**]) (access code: ), or [**Google Drive**]().

## Data preparation

DESOBA dataset include six subfolders including: ShadowImage, DeshadowedImage, InstanceMask, ShadowMask, newshadowmask, shadowparams

## Shadow Generation Baselines

### 1. Pix2Pix

- Image to image translation method. Implementation of paper "*Image-to-Image Translation with Conditional Adversarial Nets*" [[pdf]](https://arxiv.org/pdf/1611.07004.pdf).

### 2. Pix2Pix-Res

- Image to image translation method. Implementation of paper "*Image-to-Image Translation with Conditional Adversarial Nets*" [[pdf]](https://arxiv.org/pdf/1611.07004.pdf). Pix2Pix-Res is a variant of Pix2Pix whose architecture is the same as Pix2Pix but outputs the residual results.

### 3. ShadowGAN

- Image to image translation method. Implementation of paper "*ShadowGAN: Shadow synthesis for virtual objects with conditional adversarial networks*" [[pdf]](https://dc.tsinghuajournals.com/cgi/viewcontent.cgi?article=1127&context=computational-visual-media).



### 4. Mask-ShadowGAN

- Image to image translation method. Implementation of paper "*Mask-ShadowGAN: Learning to remove shadows from unpaired data*" [[pdf]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hu_Mask-ShadowGAN_Learning_to_Remove_Shadows_From_Unpaired_Data_ICCV_2019_paper.pdf).


### 5. ARShadowGAN

- Image to image translation method. Implementation of paper "*ARShadowGAN: Shadow Generative Adversarial Network for Augmented Reality in Single Light Scenes*" [[pdf]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_ARShadowGAN_Shadow_Generative_Adversarial_Network_for_Augmented_Reality_in_Single_CVPR_2020_paper.pdf).



## Test on your own data

you should provide foreground object mask and background image without foreground shadow.

- For our SGRNet method:

-- with background object mask and background shadow mask, you can modify (reseting the paths of your own data) and run,

`/src/script/SGRNet_RealComposite.sh` 

-- without background object mask and background shadow mask, you can modify (reseting the paths of your own data) and run,

`/src/script/SGRNet_RealComposite_2.sh` 

- For baselines methods:
you can run,\
`/src/script/Pix2pix_RealComposite.sh` \
`/src/script/Pix2pixRes_RealComposite.sh`\
`/src/script/ARShadowGAN_RealComposite.sh`\
`/src/script/MaskshadowGAN_RealComposite.sh`\
`/src/script/ShadowGAN_RealComposite.sh`\
to produce foreground shadow for your own composite images using pretrained baseline models.




## Test on real composite images
we provide 100 real composite images without foreground mask, you can download from [] to /DESOBA_DATASET/, and run

`/src/script/SGRNet_RealComposite.sh` 

Similarlly, you can run,\
`/src/script/Pix2pix_RealComposite.sh`\
`/src/script/Pix2pixRes_RealComposite.sh`\
`/src/script/ARShadowGAN_RealComposite.sh`\
`/src/script/MaskshadowGAN_RealComposite.sh`\
`/src/script/ShadowGAN_RealComposite.sh`\
to produce shadow for prepared real composite images using pretrained baseline models.


## Test on DESOBA dataset:
- Test on bos test images, setting `TESTDATA='--bos'` in `/src/script/SGRNet_test.sh` and run

`/script/SGRNet_test.sh` by 

- Test on bosfree test images, setting `TESTDATA='--bosfree'` in `/src/script/SGRNet_test.sh` and run

`/script/SGRNet_test.sh` 

Similarlly, you can run,\
`/src/script/Pix2pix_test.sh`\
`/src/script/Pix2pixRes_test.sh`\
`/src/script/ARShadowGAN_test.sh`\
`/src/script/MaskshadowGAN_test.sh`\
`/src/script/ShadowGAN_test.sh` \
to conduct evaluation on desoba dataset for different baseline methods


## Train on DESOBA dataset:
Considering that our SGRNet relied on shadow parameters during training phase, we need to calculate shadow parameters from paired shadow-deshadow images and we provide calculated shadow parameters in, you can also calculate shadow parameters based on tuple {shadowimage, deshadowed image, shadow mask} by refer

you can run,

`/src/script/SGRNet_train.sh`

Similarlly, you can run,\
`Pix2pix_train.sh`\
`Pix2pixRes_train.sh`\
`ARShadowGAN_train.sh`\
`MaskshadowGAN_train.sh`\
`ShadowGAN_train.sh` \
to conduct experiment on DESOBA dataset for different baseline methods.


## Experimental results on DESOBA dataset
<table class="tg">
  <tr>
    <th class="tg-0pky" align="center">Method</th>
    <th class="tg-0pky" colspan="3" align="center">BOS Test Images</th>
    <th class="tg-0pky" colspan="2" align="center">BOS-free Test Images</th>
  </tr>
  <tr>
    <th class="tg-0pky" align="center">Evaluation metric</th>
    <th class="tg-0pky" align="center">GMSE</th>
    <th class="tg-0pky" align="center">LMSE</th>
    <th class="tg-0pky" align="center">GSSIM</th>    
    <th class="tg-0pky" align="center">LSSIM</th>
    <th class="tg-0pky" align="center">GMSE</th>
    <th class="tg-0pky" align="center">LMSE</th>
    <th class="tg-0pky" align="center">GSSIM</th>    
    <th class="tg-0pky" align="center">LSSIM</th>
  </tr>
  <tr>
    <th class="tg-0pky" align="center">Pix2Pix</th>
    <th class="tg-0pky" align="center">7.659</th>
    <th class="tg-0pky" align="center">75.346</th>
    <th class="tg-0pky" align="center">0.927</th>    
    <th class="tg-0pky" align="center">0.588</th>
    <th class="tg-0pky" align="center">18.875</th>
    <th class="tg-0pky" align="center">81.444</th>
    <th class="tg-0pky" align="center">0.856</th>    
    <th class="tg-0pky" align="center">0.483</th>
  </tr>
  
  <tr>
    <th class="tg-0pky" align="center">Pix2Pix-Res</th>
    <th class="tg-0pky" align="center">5.961</th>
    <th class="tg-0pky" align="center">76.046</th>
    <th class="tg-0pky" align="center">0.971</th>    
    <th class="tg-0pky" align="center">0.612</th>
    <th class="tg-0pky" align="center">18.305</th>
    <th class="tg-0pky" align="center">81.966</th>
    <th class="tg-0pky" align="center">0.901</th>    
    <th class="tg-0pky" align="center">0.504</th>
  </tr>
  
  <tr>
    <th class="tg-0pky" align="center">ShadowGAN</th>
    <th class="tg-0pky" align="center">5.985</th>
    <th class="tg-0pky" align="center">78.413</th>
    <th class="tg-0pky" align="center">0.986</th>    
    <th class="tg-0pky" align="center">0.613</th>
    <th class="tg-0pky" align="center">19.306</th>
    <th class="tg-0pky" align="center">87.017</th>
    <th class="tg-0pky" align="center">0.918</th>    
    <th class="tg-0pky" align="center">0.498</th>
  </tr>
  
  <tr>
    <th class="tg-0pky" align="center">Mask-ShadowGAN</th>
    <th class="tg-0pky" align="center">8.287</th>
    <th class="tg-0pky" align="center">79.212</th>
    <th class="tg-0pky" align="center">0.953</th>    
    <th class="tg-0pky" align="center">0.599</th>
    <th class="tg-0pky" align="center">19.475</th>
    <th class="tg-0pky" align="center">83.457</th>
    <th class="tg-0pky" align="center">0.891</th>    
    <th class="tg-0pky" align="center">0.500</th>
  </tr>
 
 <tr>
    <th class="tg-0pky" align="center">ARShandowGAN</th>
    <th class="tg-0pky" align="center">6.481</th>
    <th class="tg-0pky" align="center">75.099</th>
    <th class="tg-0pky" align="center">0.983</th>    
    <th class="tg-0pky" align="center">0.617</th>
    <th class="tg-0pky" align="center">18.723</th>
    <th class="tg-0pky" align="center">81.272</th>
    <th class="tg-0pky" align="center">0.917</th>    
    <th class="tg-0pky" align="center">0.513</th>
  </tr>
  
  <tr>
    <th class="tg-0pky" align="center">SGRNet</th>
    <th class="tg-0pky" align="center">4.754</th>
    <th class="tg-0pky" align="center">61.762</th>
    <th class="tg-0pky" align="center">0.988</th>    
    <th class="tg-0pky" align="center">0.684</th>
    <th class="tg-0pky" align="center">15.128</th>
    <th class="tg-0pky" align="center">61.439</th>
    <th class="tg-0pky" align="center">0.927</th>    
    <th class="tg-0pky" align="center">0.555</th>
  </tr>
  
  
  
  



## Bibtex
If you find this work is useful for your research, please cite our paper using the following **BibTeX  [[arxiv](https://arxiv.org/pdf/2104.10338v1.pdf)]:**

```
@article{hong2021shadow,
  title={Shadow Generation for Composite Image in Real-world Scenes},
  author={Hong, Yan and Niu, Li and Zhang, Jianfu},
  journal={AAAI},
  year={2022}
}
```
