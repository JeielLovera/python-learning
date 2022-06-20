import cv2
import mediapipe as mp
import math
import numpy as np
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def getNeck(w,h,results,img):
  y_s=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y*h))
  y_m=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y*h))
  return int((y_s+y_m)*(1-0.1)/2) 

def getNeck2(w,h,results,img):
  y_s=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y*h))
  y_m=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y*h))
  return int((y_s+y_m)/2)

def getMidtermPointRight(w,h,results,img):
  y_w=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y*h))
  x_w=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x*w))
  x_f=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x*w))
  return int((x_w+x_f)/2),int(y_w*(1+0.03)) 

def getMidtermPointLeft(w,h,results,img):
  y_w=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y*h))
  x_w=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x*w))
  x_f=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x*w))
  return int((x_w+x_f)/2),int(y_w*(1+0.03)) 

def getBotPointRight(w,h,results,img):
  y_a=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y*h))
  x_a=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x*w))
  return int((x_a+(w/2))/2),int(y_a*(1+0.01))

def getBotPointLeft(w,h,results,img):
  y_a=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y*h))
  x_a=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x*w))
  return int((x_a+(w/2))/2),int(y_a*(1+0.01))

def getBotPointLeft2(w,h,results,img):
  y_a=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y*h))
  x_a=(int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x*w))
  return int(y_a*(1+0.01)) 

def process_person_body(file: str) -> any:
    BG_COLOR = (255, 255, 255)
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
    
      image = cv2.imread(file)
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = pose.process(image_rgb)
      h,w,_=image.shape

      if results.pose_landmarks is not None:

        #mp_drawing.draw_landmarks(image, results.pose_landmarks,
        #                          mp_pose.POSE_CONNECTIONS)

        image_no_bg = image.copy()
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        image_no_bg = np.where(condition, image_no_bg, bg_image)

        if 'frontal' in file:
          neck_y=getNeck(w,h,results,image_no_bg)

          mid_right_x,mid_right_y=getMidtermPointRight(w,h,results,image_no_bg)

          mid_left_x,mid_left_y=getMidtermPointLeft(w,h,results,image_no_bg)

          bot_right_x,bot_right_y=getBotPointRight(w,h,results,image_no_bg)

          bot_left_x,bot_left_y=getBotPointLeft(w,h,results,image_no_bg)  

          image_no_bg[:neck_y][image_no_bg[:neck_y]!=255]=255

          image_no_bg[mid_right_y:,:mid_right_x][image_no_bg[mid_right_y:,:mid_right_x]!=255]=255

          image_no_bg[mid_left_y:,mid_left_x:][image_no_bg[mid_left_y:,mid_left_x:]!=255]=255

          image_no_bg[bot_right_y:,:bot_right_x][image_no_bg[bot_right_y:,:bot_right_x]!=255]=255

          image_no_bg[bot_left_y:,bot_left_x:][image_no_bg[bot_left_y:,bot_left_x:]!=255]=255      

        if 'lateral' in file:
          neck_y=getNeck2(w,h,results,image_no_bg)
          
          bot_left_y=getBotPointLeft2(w,h,results,image_no_bg)  

          image_no_bg[:neck_y][image_no_bg[:neck_y]!=255]=255
          
          image_no_bg[bot_left_y:][image_no_bg[bot_left_y:]!=255]=255      

        image_no_bg[image_no_bg!=255]=0
        cv2.imshow("aea", image_no_bg)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return image_no_bg

process_person_body('female_0001_frontal.png')