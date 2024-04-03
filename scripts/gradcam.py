"""
author: 魔鬼面具
source: https://github.com/z1069614715/objectdetection_script/blob/master/yolo-gradcam/yolov8_heatmap.py
"""

import os
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pytorch_grad_cam import *
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
from tqdm import trange

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
np.random.seed(0)


def main():
    methods = [
        "EigenCAM",
        "EigenGradCAM",
        "GradCAM",
        "GradCAMPlusPlus",
        "HiResCAM",
        "LayerCAM",
        "RandomCAM",
        "XGradCAM",
    ]
    img_path = Path("datasets/demo_slides/flip/")
    models = [
        "runs/season/All-Season-season-3ch/weights/best.pt",
        "runs/season/All-Season-season-1ch/weights/best.pt",
        "runs/season/All-Season-season-4ch/weights/best.pt",
        "runs/season/All-Season-season-2stream/weights/best.pt",
    ]
    names = ["3ch", "1ch", "4ch", "2stream"]
    for model, name in zip(models, names):
        save_dir = Path(f"runs/gradcam/{name}")
        for method in methods:
            heatmap = yolov8_heatmap(**get_params(method, model))
            heatmap(img_path, save_dir)
    subplot()


def get_params(method="GradCAM", weight="best.pt"):
    if "2stream" in weight:  # 29, 32, 35
        layer = [29, 32, 35]
    else:  # 15, 18, 21
        layer = [15, 18, 21]

    params = {
        "weight": weight,
        "device": "cuda:0",
        "method": method,
        "layer": layer,
        "backward_type": "box",  # class, box, all
        "conf": 0.1,
        "iou": 0.25,
        "ratio": 0.02,  # 0.02-0.1
        "show_box": True,
        "renormalize": False,
    }
    return params


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class ActivationsAndGradients:
    """Class for extracting activations and
    registering gradients from targetted intermediate layers"""

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return (
            torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]],
            torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]],
            xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy(),
        )

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0])
        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()


class yolov8_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
            # if float(post_result[i].max()) < self.conf:
            #     break
            if self.ouput_type == "class" or self.ouput_type == "all":
                result.append(post_result[i].max())
            elif self.ouput_type == "box" or self.ouput_type == "all":
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)


class yolov8_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf, iou, ratio, show_box, renormalize):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt["model"].names
        model = attempt_load_weights(weight, device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()

        target = yolov8_target(backward_type, conf, ratio)
        ch = model.yaml.get("ch")
        name = method
        target_layers = [model.model[l] for l in layer]
        method = eval(method)(model, target_layers)
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(int)
        self.__dict__.update(locals())

    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf, iou_thres=self.iou)[0]
        return result

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        color = tuple(int(x) for x in color)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        # cv2.putText(
        #     img,
        #     str(name),
        #     (xmin, ymin - 5),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.8,
        #     tuple(int(x) for x in color),
        #     2,
        #     lineType=cv2.LINE_AA,
        # )
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1]
        inside every bounding boxes, and zero outside of the bounding boxes."""
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def process(self, img_path, save_path):
        # img process
        if self.ch == 4:
            RGB = cv2.imread(str(img_path))
            RGB = cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB)
            FIR = cv2.imread(str(img_path).replace("RGB", "FIR"), 0)
            FUSE = cv2.addWeighted(RGB, 0.5, cv2.cvtColor(FIR, cv2.COLOR_GRAY2RGB), 0.5, 0)
            img = cv2.merge((FIR, RGB))
        elif self.ch == 1:
            img = cv2.imread(str(img_path), 0)
        else:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = letterbox(img)[0]
        img = np.expand_dims(img, -1) if len(img.shape) < 3 else img
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError:
            return

        grayscale_cam = grayscale_cam[0, :]
        if self.ch == 4:
            img = letterbox(FUSE)[0]
            img = np.float32(img) / 255.0
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        pred = self.model(tensor)[0]
        pred = self.post_process(pred)
        if self.renormalize:
            cam_image = self.renormalize_cam_in_bounding_boxes(
                pred[:, :4].cpu().detach().numpy().astype(np.int32),
                img,
                grayscale_cam,
            )
        if self.show_box:
            for data in pred:
                data = data.cpu().detach().numpy()
                cam_image = self.draw_detections(
                    data[:4],
                    self.colors[int(data[4:].argmax())],
                    f"{self.model_names[int(data[4:].argmax())]} {float(data[4:].max()):.2f}",
                    cam_image,
                )

        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path / f"{img_path.stem}_{self.name}.jpg")

    def __call__(self, img_path, save_path):
        os.makedirs(save_path, exist_ok=True)

        if self.ch == 1:
            src_fps = img_path.glob("FIR/*.jpg")
        else:
            src_fps = img_path.glob("RGB/*.jpg")
        for src_fp in src_fps:
            self.process(src_fp, save_path)


def subplot(path=Path("runs/gradcam")):
    # Define the folder names and the common image file name
    folder_names = [x for x in path.glob("*") if x.is_dir()]
    image_paths = [x for x in list(folder_names)[0].glob("*.jpg") if x.is_file()]

    # Iterate over each folder and image
    for image_path in image_paths:
        # Create a figure with subplots
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        for i, folder_name in enumerate(folder_names):

            # Open the image
            image = Image.open(folder_name / image_path.name)
            # Convert the image to an array and display it in the subplot
            axs[i].imshow(image)
            axs[i].set_title(str(folder_name.name))
            axs[i].axis("off")  # Hide the axis
        # Adjust layout and display the plot
        fig.suptitle(image_path.name, size=20)
        fig.tight_layout()
        fig.savefig(path / f"{image_path.name}")
        print(path / f"{image_path.name}")


if __name__ == "__main__":
    main()
