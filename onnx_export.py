# Directly run this script to generate onnx model
"""
Go to /home/jingmliang/Projects/ObjectDetection/RF-DETR/rfdetr/config.py
Change RFDETRBase - resolution: int = 560
(DON'T DO THAT: will make things worse)
"""
from rfdetr import RFDETRBase

BEST_WEIGHTS = "/home/jingmliang/Projects/ObjectDetection/RF-DETR/rfdetr/best_models/checkpoint_best_total.pth"

model = RFDETRBase(pretrain_weights=BEST_WEIGHTS)

model.export()