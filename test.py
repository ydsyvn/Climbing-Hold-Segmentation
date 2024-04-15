import streamlit as st
import cv2
import math
import random
import os, json
import numpy as np
import torch, torchvision
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode

# Load the Detectron2 model
cfg = get_cfg()
cfg.merge_from_file("model\config.yaml")
cfg.MODEL.WEIGHTS = "model\model_final (1).pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold
cfg.MODEL.DEVICE = 'cpu'  # Set the device to 'cpu'
cfg.freeze()
predictor = DefaultPredictor(cfg)

def generate_boulder_route(detectedHolds, image_size):
    max_attempts = 5  # Maximum number of attempts to generate a valid route

    for _ in range(max_attempts):
        # Sort the holds by y-coordinate (top to bottom)
        sortedHolds = sorted(detectedHolds, key=lambda x: x[1], reverse=True)

        # Select a random hold from the lower quarter of the image
        lower_index = int(len(sortedHolds) * 0.2)
        upper_index = int(len(sortedHolds) * 0.3)
        start_index = random.randint(lower_index, upper_index)
        startHold = sortedHolds[start_index]

        # Select holds for the route, ensuring they're not too far apart
        routeHolds = [startHold]
        maxDistance = int(image_size[1] * 0.2)  # 20% of the image height
        minDistance = int(image_size[1] * 0.1)  # 10% of the image height
        minVerticalDistance = int(image_size[1] * 0.05)
        prevHold = startHold
        for hold in sortedHolds[start_index:]:
            if distance(prevHold, hold) >= minDistance and distance(prevHold, hold) <= maxDistance and abs(hold[1] - prevHold[1]) >= minVerticalDistance:
                routeHolds.append(hold)
                prevHold = hold

        # Check if the last hold is in the top 20% of the image
        if routeHolds[-1][1] < image_size[1] * 0.2:
            # The route ends at the top, return it
            break
    else:
        # If the loop completes without finding a valid route, return an empty route
        return [], []

    # Connect the route holds
    routeLines = []
    for i in range(1, len(routeHolds)):
        prevHold = routeHolds[i-1]
        currHold = routeHolds[i]
        routeLines.append(((prevHold[0], prevHold[1]), (currHold[0], currHold[1])))

    return routeHolds, routeLines

def distance(hold1, hold2):
    return math.sqrt((hold1[0] - hold2[0])**2 + (hold1[1] - hold2[1])**2)

def draw_boulder_route(image, routeHolds, routeLines):
    # Draw the detected holds
    for hold in routeHolds:
        x, y = hold
        cv2.circle(image, (int(x), int(y)), 10, (0, 255, 0), 2)

    # Draw the route lines
    for line in routeLines:
        (x1, y1), (x2, y2) = line
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    return image

def main():
    st.set_page_config(layout="wide")  # Set the page layout to wide

    col1, col2 = st.columns(2)
    uploaded = False

    with col1:
        st.title("Climbing Hold Instance Segmentation")

        # Allow the user to upload an image
        uploaded_file = st.file_uploader("Choose an image of a climbing wall", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            uploaded = True
            # Load the uploaded image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            height, width, channels = image.shape
            size = (width, height)

            # Get the model's predictions
            outputs = predictor(image)
            instances = outputs["instances"]

            # Extract the bounding boxes of the detected climbing holds
            detectedHolds = []
            for box in instances.pred_boxes:
                x1, y1, x2, y2 = box.tolist()
                detectedHolds.append(((x1 + x2) / 2, (y1 + y2) / 2))

            # Draw the model's predictions on the image
            v = Visualizer(image[:, :, ::-1], metadata=None, instance_mode=ColorMode.SEGMENTATION)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            output_image = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGB2BGR)

            # Add a button to trigger the boulder generation
            if st.button("Generate Climbing Route"):
                # Generate the boulder route and draw it
                routeHolds, routeLines = generate_boulder_route(detectedHolds, size)
                output_image = draw_boulder_route(output_image, routeHolds, routeLines)

    with col2:
        st.title("Prediction")
        if uploaded:
            # Display the updated image with the generated boulder route
            st.image(output_image, channels="BGR")
        else:
            st.subheader("Please upload image")

if __name__ == "__main__":
    main()