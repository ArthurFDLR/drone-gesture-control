import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow import keras
from keras.utils.data_utils import get_file
from tensorflow.python.saved_model import tag_constants

import json
import numpy as np
from pathlib import Path

#from pose_classification_kit.datasets import bodyDataset, BODY18

MODELS_PATH = Path('.') / "drone-gesture-control" / "models"


def install_pck():
    # Install Pose classification model

    #for key,val in bodyDataset(testSplit = .3, bodyModel = BODY18).items():
    #    val = np.array(val)
    #    exec(key + '=val')
    #    print(key+':', val.shape)

    model_keras = keras.models.load_model(str(MODELS_PATH / "Robust_BODY18.h5"))

    #with open('Robust_BODY18_Info.json') as f:
    #    model_labels = np.array(json.load(f)['labels'])
    #assert np.array_equiv(model_labels, labels)

    model_keras.save(str(MODELS_PATH / "Robust_BODY18"))

    params_16 = tf.experimental.tensorrt.ConversionParams(
        precision_mode='FP16',
        max_workspace_size_bytes=(1<<25),
        maximum_cached_engines=64
    )

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=str(MODELS_PATH / "Robust_BODY18"),
        conversion_params=params_16)

    # Converter method used to partition and optimize TensorRT compatible segments
    converter.convert()

    # Optionally, build TensorRT engines before deployment to save time at runtime
    # Note that this is GPU specific, and as a rule of thumb, we recommend building at runtime
    def input_fn():
        yield [np.random.normal(size=(1, 18, 2)).astype(np.float32)]

    converter.build(input_fn=input_fn)

    # Save the model to the disk 
    converter.save(str(MODELS_PATH / "Robust_BODY18_TRT"))


# Install pose estimation model https://github.com/NVIDIA-AI-IOT/trt_pose

import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt

def install_trtpose():

    with open(str(MODELS_PATH / "human_pose.json"), 'r') as f:
        human_pose = json.load(f)

    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])

    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

    MODEL_WEIGHTS = MODELS_PATH / 'resnet18_baseline_att_224x224_A_epoch_249.pth'

    model.load_state_dict(torch.load(str(MODEL_WEIGHTS)))

    WIDTH = 224
    HEIGHT = 224

    data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

    OPTIMIZED_MODEL = MODELS_PATH / 'resnet18_baseline_att_224x224_trt.pth'

    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), str(OPTIMIZED_MODEL))

install_trtpose()
install_pck()