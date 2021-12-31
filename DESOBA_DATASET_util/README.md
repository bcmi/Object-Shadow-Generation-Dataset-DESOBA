# visualization of training/test synthetic-shadow image pairs
cd DESOBA_DATASET
* visualizing bos test pairs
python Vis_Desoba_Dataset.py --isTrain 0 --bos --data_root
*visualizing bosfree pairs
python Vis_Desoba_Dataset.py --isTrain 0 --bosfrees --data_root


# producing real composite images
cd DESOBA_DATASET
* one foregound object
python Vis_RealCompositionImages_fromDesoba_Dataset.py --isTrain 0  --foreground_object_num 1 --loadSize 256
* two foreground object
python Vis_RealCompositionImages_fromDesoba_Dataset.py --isTrain 0  --foreground_object_num 2 --loadSize 256


# producing test pairs for testing
cd DESOBA_DATASET
* bos test pairs
python Store_TestPairs_Desoba_Dataset.py --bos
* bosfree pairs
python Store_TestPairs_Desoba_Dataset.py --bosfree
