

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model

def getModel(config_file="configs/pretrain/glip_Swin_T_O365_GoldG.yaml"):
    cfg.merge_from_file(config_file)
    model = build_detection_model(cfg)
    return model


model=getModel()
print(model)
with open('model_arch.log','w') as f:
    f.write(str(model))

'''
V: swin+fpn
            +VLDyHead
L: bert_enc
'''

