# Problem Definition

**Object Shadow Generation** is to deal with the shadow inconsistency between the foreground object and the background in a composite image, that is, generating plausible shadow for the foreground object according to background information, to make the composite image more realistic.

Our dataset **DESOBA** is a synthesized dataset for Object Shadow Generation. We build our dataset on the basis of Shadow-OBject Association dataset [SOBA](https://github.com/stevewongv/InstanceShadowDetection),  which  collects  real-world  images  in  complex  scenes  and  provides annotated masks for object-shadow pairs.  Based on SOBA dataset, we remove all the shadows to construct our DEshadowed Shadow-OBject Association (DESOBA) dataset, which can be used for shadow generation task and other shadow-related tasks as well. 


<img src='Examples/task_intro.png' align="center" width=800>

**Illustration of DESOBA dataset construction (top flow):**  Given a ground-truth target image I<sub>g</sub>, we manually remove all shadows to produce a deshadowed image I<sub>d</sub>. Then, we randomly select a foreground object in I<sub>g</sub>, and replace its shadow area with the counterpart in I<sub>d</sub> to synthesize a composite image I<sub>c</sub> without foreground shadow. I<sub>c</sub> and I<sub>g</sub> form a pair of input composite image and ground-truth target image. 

**Illustration of shadow generation task (bottom flow):**  Given I<sub>c</sub> and its foreground object mask M<sub>fo</sub>, we aim to reconstruct the target image I<sub>g</sub> with foreground shadow.

# Our DESOBA Dataset

**Our DESOBA dataset contains 839 training images with totally 2,995 object-shadow pairs and 160 test images with totally 624 object-shadow pairs.** Note that we discard one complex training image from SOBA. During testing, we ignore 9 object-shadow pairs with too small shadows after the whole image is resized to 256×256, resulting in 615 test image pairs.The DESOBA dataset is provided in [**Baidu Cloud**](https://pan.baidu.com/s/1zKCMTXPJcYPbOoHEHcSPkQ) (access code: 1234), or [**Google Drive**](https://drive.google.com/drive/folders/1aB3cQi6YZeg31hSOQMPI5KbETQQmqq4f?usp=drive_link).
 
 <img src='Examples/dataset-samples.png' align="center" width=1024>

We  refer  to  the  test  image pairs with Background Object-Shadow (BOS) pairs as **BOS test image pairs** (from left to right in the left subfigure: synthetic composite image, foreground object mask, background object mask,background shadow mask, and ground-truth target image), and the remaining ones as **BOS-free test image pairs** (from left to right in the right subfigure: synthetic composite image, foreground object mask, and ground-truth target image). 

## Extended to DESOBAv2

We have extended our DESOBA dataset to [DESOBAv2](https://github.com/bcmi/Object-Shadow-Generation-Dataset-DESOBAv2) with more images and object-shadow pairs.  In  [DESOBAv2](https://github.com/bcmi/Object-Shadow-Generation-Dataset-DESOBAv2),  we also propose a diffusion-based method SGDiffusion, which can achieve much better results. 


## Visualizing training/testing pairs of DESOBA dataset

`cd ./DESOBA_DATASET_util`

- Visualizing training image pairs (839 train images with 11509 pairs):
```bash
python Vis_Desoba_Dataset --serial_batches --isTrain 1
```
Training image pairs are stored in /DESOBA_DATASET/TrainTestVisualization/train/.

- Visualizing BOS-free test image pairs (34 BOS-free test images with 34 pairs):
```bash
python Vis_Desoba_Dataset --serial_batches --isTrain 0 --bosfree
```
BOS-free test image pairs are stored in /DESOBA_DATASET/TrainTestVisualization/train/test_bosfree.

- Visualizing BOS test image pairs (126 BOS test images with 581 pairs):
```bash
python Vis_Desoba_Dataset --serial_batches --isTrain 0 --bos
```
BOS test image pairs are stored in /DESOBA_DATASET/TrainTestVisualization/train/test_bos.

We show some examples of training/testing tuples below:

<img src='/data_processing/Visualization_Examples/9.png' align="center" width=800>
<img src='/data_processing/Visualization_Examples/5.png' align="center" width=800>
<img src='/data_processing/Visualization_Examples/6.png' align="center" width=800>
<img src='/data_processing/Visualization_Examples/12.png' align="center" width=800>

**From left to right:** synthetic composite image without foreground shadow, target image with foreground shadow, foreground object mask, foreground shadow mask, background object mask, and background shadow mask.


## Producing real composite images from test images of DESOBA

`cd ./DESOBA_DATASET_util`

- Producing real composite images with one foreground object:
```bash
python Vis_RealCompositionImages_fromDesoba_Dataset.py --foreground_object_num 1
```
Real composite images with one foreground object are stored in /DESOBA_DATASET/CompositeImages/1_ForegroundObject/.

- Producing real composite images with two foreground objects: 
```bash
python Vis_RealCompositionImages_fromDesoba_Dataset.py --foreground_object_num 2
```
Real composite images with two foreground objects are stored in /DESOBA_DATASET/CompositeImages/2_ForegroundObject.

We show a real composite image with one foreground object and a real composite image with two foreground objects below:

<img src='/Examples/real_composite_samples.png' align="center" width=800>

**From left to right:** synthetic composite image without foreground shadow,  foreground object mask, background object mask, background shadow mask.

To evaluate the effectiveness of different methods in real scenarios, we prepare 100 real composite images including 74 images with one foreground object and 26 images with two foreground objects. We provide 100 real composite images in  [**Baidu Cloud**](https://pan.baidu.com/s/1HAYpUefHSI7yWGzIVwT9Rg) (access code: yy77), or [**Google Drive**](https://drive.google.com/drive/folders/1aB3cQi6YZeg31hSOQMPI5KbETQQmqq4f?usp=drive_link).


## Dataloader preparation for your own project
We provide the code of obtaining training/testing tuples, in which each tuple contains foreground object mask, foreground shadow mask, background object mask, background shadow mask, shadow image, and synthetic composite image without foreground shadow mask. The dataloader is available in `/DESOBA_DATASET_util/data/DesobaSyntheticImageGeneration_dataset.py`, which can be used as dataloader in training phase or testing phase.



# Baselines
- Pix2Pix: Image to image translation method. Implementation of paper "*Image-to-Image Translation with Conditional Adversarial Nets*" [[pdf]](https://arxiv.org/pdf/1611.07004.pdf).

- Pix2Pix-Res: Image to image translation method. Implementation of paper "*Image-to-Image Translation with Conditional Adversarial Nets*" [[pdf]](https://arxiv.org/pdf/1611.07004.pdf). Pix2Pix-Res is a variant of Pix2Pix whose architecture is the same as Pix2Pix but outputs the residual results.

- ShadowGAN: Image to image translation method. Implementation of paper "*ShadowGAN: Shadow synthesis for virtual objects with conditional adversarial networks*" [[pdf]](https://dc.tsinghuajournals.com/cgi/viewcontent.cgi?article=1127&context=computational-visual-media).

- Mask-ShadowGAN: Image to image translation method. Implementation of paper "*Mask-ShadowGAN: Learning to remove shadows from unpaired data*" [[pdf]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hu_Mask-ShadowGAN_Learning_to_Remove_Shadows_From_Unpaired_Data_ICCV_2019_paper.pdf).

- ARShadowGAN: Image to image translation method. Implementation of paper "*ARShadowGAN: Shadow Generative Adversarial Network for Augmented Reality in Single Light Scenes*" [[pdf]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_ARShadowGAN_Shadow_Generative_Adversarial_Network_for_Augmented_Reality_in_Single_CVPR_2020_paper.pdf).
 
# Our SGRNet

Here we provide PyTorch implementation and the trained model of our SGRNet.

## Prerequisites

- Python 
- Pytorch
- PIL
- you can create a new environment by using  `pip install -r requirements.txt`

## Getting Started

### Installation

- Clone this repo:

```bash
git clone https://github.com/bcmi/Object-Shadow-Generation-Dataset-DESOBA.git
cd Object-Shadow-Generation-Dataset-DESOBA
```

- Download the DESOBA dataset from [**Baidu Cloud**](https://pan.baidu.com/s/1zKCMTXPJcYPbOoHEHcSPkQ) (access code: 1234), or [**Google Drive**](https://drive.google.com/drive/folders/1juSZ2lZSRRkSPy0G-68Zo2Qozver4KEE?usp=sharing). Save it at `/DESOBA_DATASET/`.


- Download the pretrained model of SGRNet and baseline methods from [**Baidu Cloud**](https://pan.baidu.com/s/1QWXtM58bMx4N0yHT-BJrpA) (access code:1234), or [**Google Drive**](https://drive.google.com/drive/folders/16isd7fPUHW1uaW3oGniCYZqhVve5zCN1?usp=sharing). Save it at `/TrainedModels/`.

### Data preparation

DESOBA dataset has six subfolders including: ShadowImage, DeshadowedImage, InstanceMask, ShadowMask, newshadowmask, shadowparams.


### Test on your own data

You should provide foreground object mask and background image without foreground shadow.

- For SGRNet method:

 - With background object mask and background shadow mask, you can modify (reseting the paths of your own data) and run
```bash
/src/script/SGRNet_RealComposite.sh
```
to produce foreground shadow for your own composite images using pretrained SGRNet model.

 - Without background object mask and background shadow mask, you can modify (reseting the paths of your own data) and run
```bash
/src/script/SGRNet_RealComposite_2.sh
```
to produce foreground shadow for your own composite images using pretrained SGRNet model.

- For baselines methods:
You can run
```bash
/src/script/Pix2pix_RealComposite.sh
/src/script/Pix2pixRes_RealComposite.sh
/src/script/ARShadowGAN_RealComposite.sh
/src/script/MaskshadowGAN_RealComposite.sh
/src/script/ShadowGAN_RealComposite.sh
```
to produce foreground shadow for your own composite images using pretrained baseline models, respectively.


### Test on real composite images
We provide 100 real composite images with foreground mask and you can download them to /DESOBA_DATASET/. 

- For SGRNet method:
You can run
```bash
/src/script/SGRNet_RealComposite.sh
```
to produce shadow for prepared real composite images using pretrained SGRNet model.


- For baselines methods:
You can run,
```bash
/src/script/Pix2pix_RealComposite.sh
/src/script/Pix2pixRes_RealComposite.sh
/src/script/ARShadowGAN_RealComposite.sh
/src/script/MaskshadowGAN_RealComposite.sh
/src/script/ShadowGAN_RealComposite.sh
```
to produce shadow for prepared real composite images using pretrained baseline models.


### Test on DESOBA dataset
- For SGRNet method:

 - For BOS test images, set `TESTDATA=--bos` in `/src/script/SGRNet_test.sh` and run
 ```bash
 /src/script/SGRNet_test.sh
 ```
to conduct evaluation on BOS test images from DESOBA dataset.

 - For BOS-free test images, set `TESTDATA=--bosfree` in `/src/script/SGRNet_test.sh` and run
```bash
/src/script/SGRNet_test.sh
```
to conduct evaluation on BOS-free test images from DESOBA dataset.

- For baselines methods: 

 - For BOS test images, set `TESTDATA=--bos` and run
```bash
/src/script/Pix2pix_test.sh
/src/script/Pix2pixRes_test.sh
/src/script/ARShadowGAN_test.sh
/src/script/MaskshadowGAN_test.sh
/src/script/ShadowGAN_test.sh
```
to conduct evaluation on BOS test images from DESOBA dataset for different baseline methods, respectively.

 - For BOS-free test images, set `TESTDATA=--bosfree` and run
 ```bash
/src/script/Pix2pix_test.sh
/src/script/Pix2pixRes_test.sh
/src/script/ARShadowGAN_test.sh
/src/script/MaskshadowGAN_test.sh
/src/script/ShadowGAN_test.sh
```
to conduct evaluation on BOS-free test images from DESOBA dataset for different baseline methods, respectively.

### Train on DESOBA dataset
Considering that our SGRNet relied on shadow parameters during training phase, we need to calculate shadow parameters from paired shadow-deshadow images. We provide calculated shadow parameters in DESOBA dataset. You can also calculate shadow parameters based on tuple {shadowimage, deshadowed image, shadow mask} by runing `/data_processing/compute_params_DESOBA.py`

- For SGRNet method: you can run
```bash
/src/script/SGRNet_train.sh
```
to train SGRNet model on DESOBA dataset.

- For baselines methods: you can run
```bash
/src/script/Pix2pix_train.sh
/src/script/Pix2pixRes_train.sh
/src/script/ARShadowGAN_train.sh
/src/script/MaskshadowGAN_train.sh
/src/script/ShadowGAN_train.sh
```
to train model on DESOBA dataset for different baseline methods, respectively.


# Experimental Results

## Leaderboard
Here we show the quantitative results of different methods on DESOBA test set based on Root  Mean  Square  Error  (RMSE)  and  Structural  SIMilarity index (SSIM).  Global RMSE (GRMSE) and Global SSIM (GSSIM) are  calculated  over  the  whole  image,  while  Local  RMSE (LRMSE) and Local SSIM (LSSIM) are calculated over the ground-truth foreground shadow area. The paper/code/model of shadow generation related methods are summarized in [Awesome-Object-Shadow-Generation](https://github.com/bcmi/Awesome-Object-Shadow-Generation). 

<table class="tg">
  <tr>
    <th class="tg-0pky" align="center">Method</th>
    <th class="tg-0pky" colspan="4" align="center">BOS Test Images</th>
    <th class="tg-0pky" colspan="4" align="center">BOS-free Test Images</th>
  </tr>
  <tr>
    <th class="tg-0pky" align="center">Evaluation metric</th>
    <th class="tg-0pky" align="center">GRMSE</th>
    <th class="tg-0pky" align="center">LRMSE</th>
    <th class="tg-0pky" align="center">GSSIM</th>    
    <th class="tg-0pky" align="center">LSSIM*</th>
    <th class="tg-0pky" align="center">GRMSE</th>
    <th class="tg-0pky" align="center">LRMSE</th>
    <th class="tg-0pky" align="center">GSSIM</th>    
    <th class="tg-0pky" align="center">LSSIM*</th>
  </tr>
  <tr>
    <th class="tg-0pky" align="center">Pix2Pix</th>
    <th class="tg-0pky" align="center">7.659</th>
    <th class="tg-0pky" align="center">75.346</th>
    <th class="tg-0pky" align="center">0.927</th>    
    <th class="tg-0pky" align="center">0.249</th>
    <th class="tg-0pky" align="center">18.875</th>
    <th class="tg-0pky" align="center">81.444</th>
    <th class="tg-0pky" align="center">0.856</th>    
    <th class="tg-0pky" align="center">0.110</th>
  </tr>
  
  <tr>
    <th class="tg-0pky" align="center">Pix2Pix-Res</th>
    <th class="tg-0pky" align="center">5.961</th>
    <th class="tg-0pky" align="center">76.046</th>
    <th class="tg-0pky" align="center">0.971</th>    
    <th class="tg-0pky" align="center">0.253</th>
    <th class="tg-0pky" align="center">18.305</th>
    <th class="tg-0pky" align="center">81.966</th>
    <th class="tg-0pky" align="center">0.901</th>    
    <th class="tg-0pky" align="center">0.107</th>
  </tr>
  
  <tr>
    <th class="tg-0pky" align="center">ShadowGAN</th>
    <th class="tg-0pky" align="center">5.985</th>
    <th class="tg-0pky" align="center">78.413</th>
    <th class="tg-0pky" align="center">0.986</th>    
    <th class="tg-0pky" align="center">0.240</th>
    <th class="tg-0pky" align="center">19.306</th>
    <th class="tg-0pky" align="center">87.017</th>
    <th class="tg-0pky" align="center">0.918</th>    
    <th class="tg-0pky" align="center">0.078</th>
  </tr>
  
  <tr>
    <th class="tg-0pky" align="center">Mask-ShadowGAN</th>
    <th class="tg-0pky" align="center">8.287</th>
    <th class="tg-0pky" align="center">79.212</th>
    <th class="tg-0pky" align="center">0.953</th>    
    <th class="tg-0pky" align="center">0.245</th>
    <th class="tg-0pky" align="center">19.475</th>
    <th class="tg-0pky" align="center">83.457</th>
    <th class="tg-0pky" align="center">0.891</th>    
    <th class="tg-0pky" align="center">0.109</th>
  </tr>
 
 <tr>
    <th class="tg-0pky" align="center">ARShandowGAN</th>
    <th class="tg-0pky" align="center">6.481</th>
    <th class="tg-0pky" align="center">75.099</th>
    <th class="tg-0pky" align="center">0.983</th>    
    <th class="tg-0pky" align="center">0.251</th>
    <th class="tg-0pky" align="center">18.723</th>
    <th class="tg-0pky" align="center">81.272</th>
    <th class="tg-0pky" align="center">0.917</th>    
    <th class="tg-0pky" align="center">0.109</th>
  </tr>
  
  <tr>
    <th class="tg-0pky" align="center">SGRNet</th>
    <th class="tg-0pky" align="center">4.754</th>
    <th class="tg-0pky" align="center">61.762</th>
    <th class="tg-0pky" align="center">0.988</th>    
    <th class="tg-0pky" align="center">0.380</th>
    <th class="tg-0pky" align="center">15.128</th>
    <th class="tg-0pky" align="center">61.439</th>
    <th class="tg-0pky" align="center">0.927</th>    
    <th class="tg-0pky" align="center">0.183</th>
  </tr>
</table>

<font size=2.5>**\***: Note that the LSSIM results in official AAAI paper are miscalculated due to a bug in the evaluation code util/ssim.py. We sincerely apologize for this mistake and have updated the results in arXiv. </font>

## Visualization results on DESOBA dataset 
Here we show some example results of different baselines on DESOBA dataset. More examples can be found in our main paper.
 
<img src='/Examples/compare_images.png' align="center" width=1024>
 
**From left to right:** input composite image (a), foreground object mask (b), results of Pix2Pix
(c), Pix2Pix-Res (d), ShadowGAN (e), Mask-ShadowGAN (f), ARShadowGAN (g), our SGRNet (h), ground-truth (i). The results on BOS test images are shown in row 1-2, and the results on  BOS-free  test images are shown in row 3-4).

## Visualization results on real composite images 
Below we present several results of different baselines on real composite images. The 100 real composite images could be found in [**Baidu Cloud**](https://pan.baidu.com/s/1mYMfK0fMjdmFlBkSyoIiMg) (access code: 1234), or [**Google Drive**](https://drive.google.com/drive/folders/1CspARS6nBVQhF0we-jro8N5PRnfTizxM?usp=sharing).
 
<img src='/Examples/composite_images_supp.png' align="center" width=1024>
 
**From left to right:** input composite image (a), foreground object mask (b), results of Pix2Pix (c), Pix2Pix-Res (d), ShadowGAN (e), Mask-ShadowGAN (f), ARShadowGAN (g), SGRNet (h).

# Other Resources

+ [Awesome-Object-Shadow-Generation](https://github.com/bcmi/Awesome-Object-Shadow-Generation)
+ [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Image-Composition)

# Bibtex
If you find this work is useful for your research, please cite our paper using the following **BibTeX  [[arxiv](https://arxiv.org/pdf/2104.10338.pdf)]:**

```
@article{hong2021shadow,
  title={Shadow Generation for Composite Image in Real-world Scenes},
  author={Hong, Yan and Niu, Li and Zhang, Jianfu},
  journal={AAAI},
  year={2022}
}
```
