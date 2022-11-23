### 1. 下载预训练模型
```
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_a_tiny_o365.pth 
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365.pth
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg.pth
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth
```


### 2.1 coco eval
```
python -m torch.distributed.launch --nproc_per_node=4 \
        tools/test_grounding_net.py \
        --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
        --weight /root/share/lengz/Project11/pretained_weights/GLIP/glip_tiny_model_o365_goldg.pth \
        TEST.IMS_PER_BATCH 4 SOLVER.IMS_PER_BATCH 4 \
        MODEL.DYHEAD.SCORE_AGG "MEAN" \
        TEST.EVAL_TASK detection \
        MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
        OUTPUT_DIR "output"
```

### 2.2 LVIS eval
```
python -m torch.distributed.launch --nproc_per_node=4 \
        tools/test_grounding_net.py \
        --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
        --task_config configs/lvis/minival.yaml \
        --weight /root/share/lengz/Project11/pretained_weights/GLIP/glip_tiny_model_o365_goldg.pth \
        TEST.EVAL_TASK detection \
        OUTPUT_DIR /root/share/lengz/Project11/outputs/GLIP \
        TEST.CHUNKED_EVALUATION 40  TEST.IMS_PER_BATCH 4 \
        SOLVER.IMS_PER_BATCH 4 \
        TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM 3000 \
        MODEL.RETINANET.DETECTIONS_PER_IMG 300 \
        MODEL.FCOS.DETECTIONS_PER_IMG 300 \
        MODEL.ATSS.DETECTIONS_PER_IMG 300 \
        MODEL.ROI_HEADS.DETECTIONS_PER_IMG 300
```

```
'''LVIS
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=all] = 0.246
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= -1 catIds=all] = 0.329
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= -1 catIds=all] = 0.264
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=     s | maxDets= -1 catIds=all] = 0.219
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=     m | maxDets= -1 catIds=all] = 0.348
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=     l | maxDets= -1 catIds=all] = 0.432
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  r] = 0.143
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  c] = 0.194
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  f] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=all] = 0.406
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=     s | maxDets= -1 catIds=all] = 0.310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=     m | maxDets= -1 catIds=all] = 0.512
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=     l | maxDets= -1 catIds=all] = 0.643
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=all] = 0.246
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= -1 catIds=all] = 0.329
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= -1 catIds=all] = 0.264
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=     s | maxDets= -1 catIds=all] = 0.219
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=     m | maxDets= -1 catIds=all] = 0.348
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=     l | maxDets= -1 catIds=all] = 0.432
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  r] = 0.143
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  c] = 0.194
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=  f] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= -1 catIds=all] = 0.406
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=     s | maxDets= -1 catIds=all] = 0.310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=     m | maxDets= -1 catIds=all] = 0.512
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=     l | maxDets= -1 catIds=all] = 0.643
'''
```

### 3. ODinW
```
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/AerialMaritimeDrone.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/AmericanSignLanguageLetters.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/Aquarium.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/BCCD.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/ChessPieces.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/CottontailRabbits.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/DroneControl.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/EgoHands.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/HardHatWorkers.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/MaskWearing.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/MountainDewCommercial.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/NorthAmericaMushrooms.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/OxfordPets.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/PKLot.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/Packages.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/PascalVOC.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/Raccoon.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/ShellfishOpenImages.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/ThermalCheetah.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/UnoCards.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/VehiclesOpenImages.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/WildfireSmoke.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/boggleBoards.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/brackishUnderwater.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/dice.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/openPoetryVision.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/pistols.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/plantdoc.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/pothole.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/selfdrivingCar.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/thermalDogsAndPeople.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/vector.zip
wget https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35/websiteScreenshots.zip
```


### 4. finetune on COCO
```还没跑试试
CUDA_VISIBLE_DEVICE=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py \
       --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
       --skip-test \
       MODEL.WEIGHT /root/share/lengz/Project11/pretained_weights/GLIP/glip_tiny_model_o365_goldg.pth \
       DATASETS.TRAIN '("coco_grounding_train", )' \
       MODEL.BACKBONE.FREEZE_CONV_BODY_AT -1  \
       SOLVER.IMS_PER_BATCH 4 \
       SOLVER.USE_AMP True \
       SOLVER.MAX_EPOCH 12 \
       TEST.DURING_TRAINING False \
       TEST.IMS_PER_BATCH 4 \
       SOLVER.FIND_UNUSED_PARAMETERS False \
       SOLVER.BASE_LR 0.00001 \
       SOLVER.LANG_LR 0.00001 \
       SOLVER.STEPS \(0.67,0.89\) \
       DATASETS.DISABLE_SHUFFLE True \
       MODEL.DYHEAD.SCORE_AGG "MEAN" \
       TEST.EVAL_TASK detection
```







