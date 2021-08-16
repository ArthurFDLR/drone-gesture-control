import os

os.environ["MPLCONFIGDIR"] = os.getcwd() + "/configs/"

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow import keras

import json
import numpy as np
from pathlib import Path

# Install pose estimation model https://github.com/NVIDIA-AI-IOT/trt_pose
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt

MODELS_PATH = Path(".") / "drone-gesture-control" / "models"


def install_pck():
    model_keras = keras.models.load_model(str(MODELS_PATH / "Robust_BODY18.h5"))
    model_keras.save(str(MODELS_PATH / "Robust_BODY18"))
    params = tf.experimental.tensorrt.ConversionParams(
        precision_mode="FP16",
        max_workspace_size_bytes=(1 << 25),
        maximum_cached_engines=64,
    )
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=str(MODELS_PATH / "Robust_BODY18"),
        conversion_params=params,
    )
    converter.convert()

    def input_fn():
        yield [np.random.normal(size=(1, 18, 2)).astype(np.float32)]

    converter.build(input_fn=input_fn)
    converter.save(str(MODELS_PATH / "Robust_BODY18_TRT"))


def install_trtpose():
    MODEL_WEIGHTS = MODELS_PATH / "resnet18_baseline_att_224x224_A_epoch_249.pth"
    OPTIMIZED_MODEL = MODELS_PATH / "resnet18_baseline_att_224x224_trt.pth"
    WIDTH, HEIGHT = 224, 224  # Input resolution

    with open(str(MODELS_PATH / "human_pose.json"), "r") as f:
        human_pose = json.load(f)

    num_parts = len(human_pose["keypoints"])
    num_links = len(human_pose["skeleton"])
    model = (
        trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    )
    model.load_state_dict(torch.load(str(MODEL_WEIGHTS)))
    data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
    model_trt = torch2trt.torch2trt(
        model, [data], fp16_mode=True, max_workspace_size=1 << 25
    )
    torch.save(model_trt.state_dict(), str(OPTIMIZED_MODEL))


if __name__ == "__main__":
    install_trtpose()
    install_pck()
