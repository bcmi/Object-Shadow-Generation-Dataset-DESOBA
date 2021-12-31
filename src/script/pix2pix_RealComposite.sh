
#!/bin/bash
batchs=1
GPU=0
lr=0.0002
loadSize=256
fineSize=256
L1=100
model=Pix2pix
G='RESNEXT18'
ngf=32


L_shadowrecons=10
L_imagerecons=10
L_GAN=1
Residual=0


#####network design
DISPLAY_PORT=8002
D='pixel'
lr_D=0.0002



# model infomation
checkpoint='../../TrainedModels/Pix2pix_TrainedModel/'
model_name=Basic
NAME="${model_name}_C${ngf}_D${D}_lrD${lr_D}"


#####testing for real composite images
datasetmode=ShadowGenerationDatasetInference2
# dataroot='../../DESOBA_DATASET/CompositionShadowGeneration-100/OneforegroundObject74/'
dataroot='../../DESOBA_DATASET/CompositionShadowGeneration-100/TwoforegroundObject26/'
shadowfree_path=${dataroot}'/shadowfree_img'
instance_path=${dataroot}'/foreground_object_mask'


OTHER="--no_crop --no_flip --no_rotate --serial_batches"

CMD="python ../generate.py --loadSize ${loadSize} \
    --phase test --eval
    --name ${NAME} \
    --checkpoints_dir ${checkpoint} \
    --epoch latest\
    --fineSize $fineSize --model $model\
    --batch_size $batchs --display_port ${DISPLAY_PORT}
    --display_server http://localhost
    --gpu_ids ${GPU} --lr ${lr} \
    --dataset_mode $datasetmode\
    --norm instance\
    --dataroot  ${dataroot}\
    --instance_path $instance_path\
    --shadowfree_path $shadowfree_path\
    --lambda_M1 $L_shadowrecons --lambda_I1 $L_imagerecons --lambda_GAN $L_GAN 
    --netG $G\
    --ngf $ngf
    --netD $D
    --lr_D $lr_D

    $OTHER"

echo $CMD
eval $CMD

