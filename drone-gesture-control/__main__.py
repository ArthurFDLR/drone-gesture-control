from src.video_processing import VideoProcessing
from src.drone_control import Drone
from dronekit import VehicleMode

import time, numpy as np, pathlib


CAM_ASPECT_RATIO = 16./9.
INPUT_IMG_SIZE = 224

gstream_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), "
    "width=(int){capture_width:d}, height=(int){capture_height:d}, "
    "format=(string)NV12, framerate=(fraction){framerate:d}/1 ! "
    "nvvidconv top={crop_top:d} bottom={crop_bottom:d} left={crop_left:d} right={crop_right:d} flip-method={flip_method:d} ! "
    "video/x-raw, width=(int){display_width:d}, height=(int){display_height:d}, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink".format(
        capture_width = int(INPUT_IMG_SIZE*CAM_ASPECT_RATIO),
        capture_height = INPUT_IMG_SIZE,
        framerate = 60,
        crop_top = 0,
        crop_bottom = INPUT_IMG_SIZE,
        crop_left = int(INPUT_IMG_SIZE*(CAM_ASPECT_RATIO-1)/2),
        crop_right = int(INPUT_IMG_SIZE*(CAM_ASPECT_RATIO+1)/2),
        flip_method = 0,
        display_width = INPUT_IMG_SIZE,
        display_height = INPUT_IMG_SIZE,
    )
)

print(gstream_pipeline)

MODELS_PATH = pathlib.Path('.') / "drone-gesture-control" / "models"

video_processing = VideoProcessing(
    gstream_def=gstream_pipeline,
    estimation_mode_path=str(MODELS_PATH/'resnet18_baseline_att_224x224_trt.pth'),
    classification_mode_path=str(MODELS_PATH/'Robust_BODY18_TRT'),
    topology_path=str(MODELS_PATH/'human_pose.json'),
    labels_path=str(MODELS_PATH/'Robust_BODY18_info.json'),
)

drone = Drone(serial_address='/dev/ttyTHS1', baud=57600)
THRESHOLD_ALT = 0.3 
last_label = video_processing.get_pose()
print('\n### RUNNING ###\n')

try:
    while True:
        latest_label = video_processing.get_pose()
        if latest_label == last_label:
            label = None
        else:
            label = latest_label
            last_label = latest_label

        if label and (drone.vehicle.mode.name=="GUIDED"):

            if label == "T":
                if not drone.vehicle.armed: # Arm
                    drone.vehicle.armed = True
                else:
                    if drone.vehicle.location.global_relative_frame.alt < THRESHOLD_ALT: # Disarm if landed
                        drone.vehicle.armed = False
            
            elif (label == "Traffic_AllStop") and drone.vehicle.armed:
                if drone.vehicle.location.global_relative_frame.alt < THRESHOLD_ALT:  # Take off if landed
                    takeoff_alt = 1.8
                    drone.vehicle.simple_takeoff(takeoff_alt) # Take off at two meters
                    while drone.vehicle.location.global_relative_frame.alt<(takeoff_alt - THRESHOLD_ALT): # Wait to reach altitude
                        time.sleep(.3)
                        #print("Altitude: ", vehicle.location.global_relative_frame.alt)
                else:
                    if drone.vehicle.location.global_relative_frame.alt > THRESHOLD_ALT:
                        drone.vehicle.mode = VehicleMode("LAND")
            
            # Go right
            elif (label == "Traffic_RightTurn") and drone.vehicle.armed:
                x, y = 0., 2.  #meters
                yaw = drone.vehicle.attitude.yaw
                drone.send_global_velocity(
                    x*np.cos(yaw) - y*np.sin(yaw),
                    x*np.sin(yaw) + y*np.cos(yaw),
                    0,
                    2
                )
                drone.send_global_velocity(0,0,0,1)
            
            # Go left
            elif (label == "Traffic_LeftTurn") and drone.vehicle.armed:
                x, y = 0., -2.  #meters
                yaw = drone.vehicle.attitude.yaw
                drone.send_global_velocity(
                    x*np.cos(yaw) - y*np.sin(yaw),
                    x*np.sin(yaw) + y*np.cos(yaw),
                    0,
                    2
                )
                drone.send_global_velocity(0,0,0,1)
            
            # Go back
            elif (label == "Traffic_BackFrontStop") and drone.vehicle.armed:
                x, y = -2.0, 0.  #meters
                yaw = drone.vehicle.attitude.yaw
                drone.send_global_velocity(
                    x*np.cos(yaw) - y*np.sin(yaw),
                    x*np.sin(yaw) + y*np.cos(yaw),
                    0,
                    2
                )
                drone.send_global_velocity(0,0,0,1)
            
            # Go front
            elif (label == "Traffic_FrontStop" or label == "Stand_RightArmRaised") and drone.vehicle.armed:
                x, y = 2.0, 0.  #meters
                yaw = drone.vehicle.attitude.yaw
                drone.send_global_velocity(
                    x*np.cos(yaw) - y*np.sin(yaw),
                    x*np.sin(yaw) + y*np.cos(yaw),
                    0,
                    2
                )
                drone.send_global_velocity(0,0,0,1)
            
            elif (label == "Yoga_UpwardSalute") and drone.vehicle.armed:
                drone.vehicle.mode = VehicleMode("RTL") 
            
            #elif label == "rotate right":
            #    drone.condition_yaw(45,relative=True)
            
            #elif label == "rotate left":
            #    drone.condition_yaw(315,relative=True)

        time.sleep(.25)
        print('FPS: {fps:.2f}\tCurrent label: {lab1}\tSend order: {lab2}'.format(fps=video_processing.get_fps(), lab1=latest_label, lab2=label).ljust(80)[:80], end='\r')
except:
    video_processing.cap.release()
    video_processing.video_recording.release()

    print('Video processing stopped')