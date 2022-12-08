
# 根据coco的结果，生成伪标签

import json
import os
import torch
from torchvision.ops import nms
from tqdm import tqdm
import numpy as np

def main(mode='val',min_score=0.5,nms_iou_thl=0.5):
    bbox_json_path = f'/root/workspace/Project/GLIP/output/eval/glip_tiny_model_o365_goldg/inference/coco_2017_{mode}/bbox.json'
    prediction_path = f'/root/workspace/Project/GLIP/output/eval/glip_tiny_model_o365_goldg/inference/coco_2017_{mode}/predictions.pth'
    
    bbox_json = json.load(open(bbox_json_path,'r'))                 # 388679个预测框，每张图1-100个，xywh格式，恢复到了原图尺寸
    prediction = torch.load(prediction_path,map_location='cpu')     # 5000张图的每个图的预测bboxlist，xyxy格式，resize之后的尺寸
    
    coco = json.load(open(f'/root/workspace/det_datasets/coco/annotations/instances_{mode}2017.json','r'))
    
    out_json = dict(info=coco['info'],licenses=coco['licenses'],images=coco['images'],annotations=[],categories=coco['categories'])
    '''
    每条annotation有：segmentation area iscrowd image_id bbox category_id id 共7个字段，我们在bbox_json中挑选出score大于0.5的预测
    作为伪标，通过nms处理一下。如果没有大于0.5的score，则选择score最大的哪一个预测作为伪标。
    '''
    p = 0       # bbox_json当前指针
    for i in tqdm(range(len(prediction))):
        ann = dict()
        bboxes_list = []
        scores_list = []
        predi=prediction[i]
        scores = predi.extra_fields['scores']
        num_boxes = len(scores)
        bboxes = bbox_json[p:p+num_boxes]
        p+=num_boxes
        sorted(bboxes, key = lambda x:x['score'],reverse=True)
        bboxes_list.append(bboxes[0]['bbox'])
        scores_list.append(bboxes[0]['score'])
        for j in range(1,len(bboxes)):
            if bboxes[j]['score']>min_score:
                bboxes_list.append(bboxes[j]['bbox'])
                scores_list.append(bboxes[j]['score'])
        
        bboxes_tensor = torch.tensor(bboxes_list)
        bboxes_tensor[:,2] += bboxes_tensor[:,0]        # xywh转xyxy
        bboxes_tensor[:,3] += bboxes_tensor[:,1]        # xywh转xyxy
        scores_tensor = torch.tensor(scores_list)
        keep = nms(bboxes_tensor,scores_tensor,iou_threshold=nms_iou_thl)       # xyxy
        # bboxes_after_nms = bboxes_tensor[keep]
        scores_after_nms = scores_tensor[keep]
        assert len(scores_after_nms)>=1
        for xx in keep:
            bbox = bboxes[xx]['bbox']
            area = bbox[3]*bbox[2]
            image_id = bboxes[xx]['image_id']
            category_id = bboxes[xx]['category_id']
            id = len(out_json['annotations'])+100000
            ann = dict(segmentation=bbox,area=area,iscrowd=0,image_id=image_id,bbox=bbox,category_id=category_id,id=id)
            out_json['annotations'].append(ann)
        
    json.dump(out_json,open(f"/root/workspace/det_datasets/coco/pse_annotations/pse_{mode}2017_{min_score}_{nms_iou_thl}.json",'w'))
    print("ok!")
    
    
    return


# main(mode='train')
# main(mode='train',min_score=0.9,nms_iou_thl=0.5)
# main(mode='train',min_score=0.6,nms_iou_thl=0.5)
# main(mode='train',min_score=0.7,nms_iou_thl=0.5)
# main(mode='train',min_score=0.8,nms_iou_thl=0.5)
# main(mode='train',min_score=0.4,nms_iou_thl=0.5)
# main(mode='train',min_score=0.3,nms_iou_thl=0.5)
# main(mode='train',min_score=0.2,nms_iou_thl=0.5)
# main(mode='train',min_score=0.1,nms_iou_thl=0.5)
# main(mode='train',min_score=0.05,nms_iou_thl=0.5)
# main(mode='train',min_score=0.01,nms_iou_thl=0.5)
# main(mode='train',min_score=0.05,nms_iou_thl=0.9)
# main(mode='train',min_score=0.05,nms_iou_thl=0.7)
# main(mode='train',min_score=0.05,nms_iou_thl=0.3)
main(mode='train',min_score=0.05,nms_iou_thl=0.4)
main(mode='train',min_score=0.05,nms_iou_thl=0.2)
main(mode='train',min_score=0.05,nms_iou_thl=0.1)



# 通过之前的实验我们知道，伪标的质量是有讲究的，可以少一些质量太差的伪标，我们可以统计一下score和他们与gt的iou大小的分布情况，
from pycocotools.coco import COCO
# from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import matplotlib.pyplot as plt

def box_area(boxes: np.array) -> np.array:
    assert boxes.ndim == 2 and boxes.shape[-1] == 4
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def _box_inter_union(boxes1: np.array, boxes2: np.array):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter
    return inter, union

def box_iou(boxes1: np.array, boxes2: np.array) -> np.array:
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou

def scoreVsIou(mode='val',min_score=0.5,nms_iou_thl=0.5):
    bbox_json_path = f'/root/workspace/Project/GLIP/output/eval/glip_tiny_model_o365_goldg/inference/coco_2017_{mode}/bbox.json'
    prediction_path = f'/root/workspace/Project/GLIP/output/eval/glip_tiny_model_o365_goldg/inference/coco_2017_{mode}/predictions.pth'
    
    bbox_json = json.load(open(bbox_json_path,'r'))                 # 388679个预测框，每张图1-100个，xywh格式，恢复到了原图尺寸
    prediction = torch.load(prediction_path,map_location='cpu')     # 5000张图的每个图的预测bboxlist，xyxy格式，resize之后的尺寸
    
    # coco = json.load(open(f'/root/workspace/det_datasets/coco/annotations/instances_{mode}2017.json','r'))
    coco = COCO(annotation_file=f'/root/workspace/det_datasets/coco/annotations/instances_{mode}2017.json')
    # out_json = dict(info=coco['info'],licenses=coco['licenses'],images=coco['images'],annotations=[],categories=coco['categories'])

    p = 0       # bbox_json当前指针
    keep_ious = []
    keep_scores = []
    for i in tqdm(range(len(prediction))):
        ann = dict()
        bboxes_list = []
        scores_list = []
        predi=prediction[i]
        scores = predi.extra_fields['scores']
        num_boxes = len(scores)
        bboxes = bbox_json[p:p+num_boxes]
        p+=num_boxes
        sorted(bboxes, key = lambda x:x['score'],reverse=True)
        bboxes_list.append(bboxes[0]['bbox'])
        scores_list.append(bboxes[0]['score'])
        image_id = bboxes[0]['image_id']
        for j in range(1,len(bboxes)):
            if bboxes[j]['score']>min_score:
                bboxes_list.append(bboxes[j]['bbox'])
                scores_list.append(bboxes[j]['score'])
        
        bboxes_tensor = torch.tensor(bboxes_list)
        bboxes_tensor[:,2] += bboxes_tensor[:,0]        # xywh转xyxy
        bboxes_tensor[:,3] += bboxes_tensor[:,1]        # xywh转xyxy
        scores_tensor = torch.tensor(scores_list)
        keep = nms(bboxes_tensor,scores_tensor,iou_threshold=nms_iou_thl)       # xyxy
        bboxes_tensor = bboxes_tensor[keep]
        scores_tensor = scores_tensor[keep]
        # print(f'done filter(thl={min_score}) and nms(thl={nms_iou_thl})')
        anns = coco.imgToAnns[image_id]
        gt_boxes=[]
        for ann in anns:
            bx = ann['bbox']
            bx[2]+=bx[0]
            bx[3]+=bx[1]
            gt_boxes.append(bx)
        gt_boxes = torch.tensor(gt_boxes)
        if len(anns)!=0:
            ious = box_iou(bboxes_tensor,gt_boxes)      #[N, M], N预测，M Gt
            max_iou, _ = torch.max(ious,dim=-1)
        else:
            max_iou = torch.zeros_like(scores_tensor)
        keep_ious.extend(max_iou.tolist())
        keep_scores.extend(scores_tensor.tolist())
        
    corr = np.corrcoef(np.array(keep_ious),np.array(keep_scores))
    print(corr[0,1])
    # plt.scatter(keep_ious,All_scores,s=1)
    # plt.savefig("/root/workspace/Project/GLIP/tools/lz_tools/iou_score.png")
    
    
    

# scoreVsIou()





