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
from detectron2.utils.visualizer import Visualizer, ColorMode

# print(torch.__version__, torch.cuda.is_available())
# assert torch.__version__.startswith("1.8")
setup_logger()

st.title('Climbing hold segmentation')

# showing image
im = cv2.imread("images/img26.jpg")
st.image(im, channels="BGR")

#register_coco_instances("train_dataset", {}, "train.json", "segmentation images")
train_metadata = MetadataCatalog.get("train_dataset")

cfg = get_cfg()
cfg.merge_from_file("model\config.yaml")
cfg.MODEL.WEIGHTS = "model\model_final (1).pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # set a custom testing threshold
cfg.MODEL.DEVICE = 'cpu'  # Set the device to 'cpu'
cfg.freeze()
predictor = DefaultPredictor(cfg)

outputs = predictor(im)

st.write('Using Vizualizer to draw the predictions on Image')
v = Visualizer(im[:, :, ::-1], metadata=train_metadata, instance_mode=ColorMode.SEGMENTATION)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

st.image(out.get_image()[:, :, ::-1])
