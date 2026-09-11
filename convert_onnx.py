'''
Converter Application to export pytorch model as onnx
'''

import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='yolo')
parser.add_argument("--load_path", type=str)
parser.add_argument("--save_path", type=str)
parser.add_argument("--input", type=str)  #example image path
parser.add_argument("--inames", type=str, default='image')
parser.add_argument("--onames", type=str)
args = parser.parse_args()

load_path = args.load_path
save_path = args.save_path
example_input = cv2.imread(args.input)

class torchHandle:
    def __init__(self):
        import torch
        self.torch = torch

    def load_model(self):
        #Load model
        model = self.torch.load(load_path)
        model.eval()
        return model

    def export_model(self, model):
        self.torch.onnx.export(model, example_input, input_names=args.inames, output_names=args.onames,optimize=True,dynamo=True)
    
class yoloHandle:
    def __init__(self):
        pass
        
    def load_model(self, path):
        from ultralytics import YOLO
        model = YOLO(path)
        model.eval()
        return model
    
    def export_model(self, model):
        model.export(format='onnx', optimize=True, imgsz=640)

handle_selection = {'yolo': yoloHandle(), 'torch': torchHandle()}
handle = handle_selection.get(args.model, 'yolo')
model = handle.load_model(load_path)

handle.export_model(model)