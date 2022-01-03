## Visualizing training/testing pairs of DESOBA dataset

`cd ./DESOBA_DATASET_util`

- visualizing train pairs (839 train images with 11509 pairs):
```bash
python Vis_Desoba_Dataset --serial_batches --isTrain 1
```
train pairs are stored in /DESOBA_DATASET/TrainTestVisualization/train/

- visulizing test bosfree pairs (34 test bosfree images with 34 pairs):
```bash
python Vis_Desoba_Dataset --serial_batches --isTrain 0 --bosfree
```
test bosfree pairs are store in /DESOBA_DATASET/TrainTestVisualization/train/test_bosfree

- visulizing test bos pairs (126 test bos images with 581 pairs):
```bash
python Vis_Desoba_Dataset --serial_batches --isTrain 0 --bos
```
test bos pairs are stored in /DESOBA_DATASET/TrainTestVisualization/train/test_bos

We show some examples of training/testing tuples in below:
<img src='/data_processing/Visualization_Examples/9.png' align="center" width=1024>
<img src='/data_processing/Visualization_Examples/5.png' align="center" width=1024>
<img src='/data_processing/Visualization_Examples/6.png' align="center" width=1024>
<img src='/data_processing/Visualization_Examples/12.png' align="center" width=1024>
from left to right: synthetic composite image without foreground shadow, target image with foreground shadow, foreground object mask, foreground shadow mask, background object mask, and background shadow mask.


## Producing real composite images from test images of DESOBA

`cd ./DESOBA_DATASET_util`

- producing real composite images with one foreground object:
```bash
python Vis_RealCompositionImages_fromDesoba_Dataset.py --foreground_object_num 1
```
real composite images with one foreground object are store in /DESOBA_DATASET/CompositeImages/1_ForegroundObject/

- producing real composite images with two foreground objects: 
```bash
python Vis_RealCompositionImages_fromDesoba_Dataset.py --foreground_object_num 2
```

real composite images with one foreground object are stored in /DESOBA_DATASET/CompositeImages/2_ForegroundObject/

We show a real composite image with one foreground object and a real composite image with two foreground objects below:

<img src='/Examples/real_composite_samples.png' align="center" width=800>
from left to right: synthetic composite image without foreground shadow,  foreground object mask, background object mask, and background shadow mask.



## Producing test pairs of DESOBA dataset for testing

`cd ./DESOBA_DATASET_util`

- bos test pairs
```bash
python Store_TestPairs_Desoba_Dataset.py --bos
```
BOS test pairs are stored in /DESOBA_DATASET/TestSplit/bos


- bosfree pairs
```bash
python Store_TestPairs_Desoba_Dataset.py --bosfree
```
BOS-free test pairs are stored in /DESOBA_DATASET/TestSplit/bosfee
