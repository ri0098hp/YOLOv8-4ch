"""
base : 魔鬼面具 https://github.com/z1069614715/objectdetection_script/blob/master/yolo-gradcam/yolov8_heatmap.py
mod  : okuda
"""

import os
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import *  # noqa
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
from tqdm import trange

from ultralytics.cfg import check_dict_alignment, parse_key_value_pair
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.checks import check_suffix
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
np.random.seed(0)


def gradcam(args=""):
    params = get_params(args)
    heatmap = yolov8_heatmap(**params)
    heatmap()
    print("Recommend layer=[29, 32, 35] in 2stream model.")


def get_params(args):
    params = {
        "source": "ultralytics/assets/bus.jpg",
        "project": "runs/gradcam",
        "name": "",
        "backward_type": "class",  # class, box, all
        "device": "cuda:0",
        "conf": 0.1,
        "iou": 0.25,
        "ratio": 0.02,  # 0.02-0.1
        "model": "yolov8n.pt",
        "method": "XGradCAM",
        "layer": [15, 18, 21],  # [15, 18, 21] in base, [29, 32, 35] in 2stream
        "renormalize": False,
        "show_box": True,
        "only": "",  # input only RGB or FIR in 4-channel model
    }
    if not args:
        args = sys.argv
    for arg in args:
        if arg.startswith("--"):
            print(f"WARNING ⚠️ '{arg}' does not require leading dashes '--', updating to '{arg[2:]}'.")
            arg = arg[2:]
        if arg.endswith(","):
            print(f"WARNING ⚠️ '{arg}' does not require trailing comma ',', updating to '{arg[:-1]}'.")
            arg = arg[:-1]
        if "=" in arg:
            try:
                k, v = parse_key_value_pair(arg)
                params[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(params, {arg: ""}, e)
    assert params["method"][-3:] == "CAM", "'method' shold be CAM name"
    params["source"] = Path(params["source"])
    params["project"] = Path(params["project"])
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


class ActivationsAndGradients(ActivationsAndGradients):
    """Class for extracting activations and
    registering gradients from targetted intermediate layers"""

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
        """
        推論スコアと推論bboxを返す
        """

        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0])
        return [[post_result, pre_post_boxes]]


class yolov8_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, x):
        post_result, pre_post_boxes = x
        result = []
        for i in trange(int(post_result.size(0) * self.ratio), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
            if self.ouput_type == "class" or self.ouput_type == "all":
                result.append(post_result[i].max())  # highest score in bbox
            elif self.ouput_type == "box" or self.ouput_type == "all":
                for j in range(4):
                    result.append(pre_post_boxes[i, j])  # bbox 4 point coordinate
        return sum(result)


class yolov8_heatmap:
    def __init__(
        self,
        source,
        project,
        name,
        backward_type,
        device,
        conf,
        iou,
        ratio,
        model,
        method,
        layer,
        show_box,
        renormalize,
        only,
    ):
        device = torch.device(device)
        check_suffix(file=model, suffix=".pt")
        if not Path(model).exists():
            model = attempt_download_asset(model)  # downlaod offcial YOLOv8 weight
        ckpt = torch.load(model)
        model_names = ckpt["model"].names
        model = attempt_load_weights(model, device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()

        target = yolov8_target(backward_type, conf, ratio)
        ch = model.yaml.get("ch")
        target_layers = [model.model[l] for l in layer]
        cam = eval(method)(model, target_layers)
        cam.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(int)
        self.__dict__.update(locals())

    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf, iou_thres=self.iou)[0]
        return result

    def draw_detections(self, box, color, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        color = tuple(int(x) for x in color)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
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
            if self.only == "RGB":
                FIR = np.zeros_like(FIR)
                FUSE = cv2.addWeighted(RGB, 0.5, cv2.cvtColor(FIR, cv2.COLOR_GRAY2RGB), 0.5, 0)
            elif self.only == "FIR":
                RGB = np.zeros_like(RGB)
                FUSE = FIR
            else:
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
            grayscale_cam = self.cam(tensor, [self.target])
        except AttributeError:
            return
        except Exception as e:
            raise e
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
                cam_image = self.draw_detections(data[:4], self.colors[int(data[4:].argmax())], cam_image)

        fp_cam_image = save_path / f"{img_path.stem}_{self.method}.jpg"
        cam_image = Image.fromarray(cam_image)
        cam_image.save(fp_cam_image)
        grayscale_cam = Image.fromarray(show_cam(grayscale_cam)).convert("RGB")
        grayscale_cam.save(save_path / f"{img_path.stem}_{self.method}_map.jpg")
        print(f"cam image was saved at {fp_cam_image}")

    def __call__(self):
        # image loader

        if self.source.is_file():  # single file
            src_fps = [self.source]
        elif self.ch == 1:  # RGB-FIR folder only with FIR
            src_fps = self.source.glob("FIR/*.jpg")
        else:  # RGB-FIR folders
            src_fps = self.source.glob("RGB/*.jpg")
        for src_fp in src_fps:
            save_path = self.project / self.name if self.name else self.project / src_fp.stem
            os.makedirs(save_path, exist_ok=True)
            print("image:", src_fp)
            self.process(src_fp, save_path)


def show_cam(mask):
    """
    Tensor to Grascale
    """
    heatmap = mask / np.max(mask)
    heatmap = np.uint8(255 * heatmap)
    return plt.get_cmap("jet")(heatmap, bytes=True)


if __name__ == "__main__":
    gradcam()
