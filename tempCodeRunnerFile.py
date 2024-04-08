import streamlit as st
import torch,torchvision
# import req
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# print(torch.__version__, torch.cuda.is_available())
# assert torch.__version__.startswith("1.8")
setup_logger()

st.title('Climbing hold segmentation')
st.write('What is Detectron2?')
st.image('boulder app\images\img26.jpg')

# import some common detectron2 utilities

st.write('\n')

st.title('Testing the Zoo Model')
st.write('Test image')
im = cv2.imread("boulder app\images\img20.jpg")

# showing image
st.image('boulder app\images\img20.jpg')

#Then, we create a detectron2 config and a detectron2 `DefaultPredictor` to run inference on this image.
"""cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
# print(cfg)
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)"""

register_coco_instances("train_dataset", {}, "train.json", "segmentation images")
train_metadata = MetadataCatalog.get("train_dataset")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "model_final (1).pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # set a custom testing threshold
cfg.MODEL.DEVICE = 'cpu'  # Set the device to 'cpu'
cfg.freeze()
predictor = DefaultPredictor(cfg)

outputs = predictor(im)

st.write('Writing pred_classes/pred_boxes output')
st.write(outputs["instances"].pred_classes)
st.write(outputs["instances"].pred_boxes)

st.write('Using Vizualizer to draw the predictions on Image')
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
st.image(out.get_image()[:, :, ::-1])