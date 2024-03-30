from flask import Flask, render_template, request, jsonify
import torch
import detectron2
import cv2 as cv
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from PIL import Image, ImageDraw
import io
import base64

def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open("./"+fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

app = Flask(__name__)

# Initialize Detectron2 predictor
cfg = get_cfg()
cfg.merge_from_file("detectron2\\configs\\COCO-Detection\\faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
cfg.MODEL.DEVICE = 'cpu'  # Set model to run on CPU
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['image']
    file.save(file.filename)
    predictor = DefaultPredictor(cfg)
    im = cv.imread(file.filename)
    outputs = predictor(im)
    print(outputs)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    predicted_image = out.get_image()
    im_rgb = cv.cvtColor(predicted_image, cv.COLOR_RGB2BGR)
    predicted_filename="result_"+file.filename
    cv.imwrite(predicted_filename, im_rgb)
    opencodedbase64 = encodeImageIntoBase64(predicted_filename)
    result = {"image" : opencodedbase64.decode('utf-8') }
    return result

if __name__ == '__main__':
    app.run(debug=True)