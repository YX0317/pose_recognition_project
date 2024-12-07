import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,model_complexity=1,smooth_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
def calculate_pixel_distance(pt1,pt2):
        #Calculate pixel distance between two point
        return np.linalg.norm(np.array(pt1) - np.array(pt2))

#Define thresholds foe sitting and standing
sitting_threshold = 110 # adjust as needed
standing_threshold = 170 #adjust as needed

while cap.isOpened():
        success, image = cap.read()
        if not success:
                break
        
        #Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        #Process the image and find facial landmarks
        results = pose.process(image_rgb)
    
        #Convert the image color back so it can be displayed
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            #Convert landmarks to pixel coordinates
            left_hip_pixel = (int(left_hip.x * image.shape[1]),
                              int(left_hip.y * image.shape[0]))
            left_ankle_pixel = (int(left_ankle.x * image.shape[1]),
                              int(left_ankle.y * image.shape[0]))
            right_hip_pixel = (int(right_hip.x * image.shape[1]),
                              int(right_hip.y * image.shape[0]))
            right_ankle_pixel = (int(right_ankle.x * image.shape[1]),
                              int(right_ankle.y * image.shape[0]))
            
            #Draw lines between hip and ankle
            cv2.line(image,left_hip_pixel, left_ankle_pixel,(255,0,0),2)
            cv2.line(image,right_hip_pixel,right_ankle_pixel,(0,0,255),2)
            
            #Calculate distance in pixels
            left_leg_lenght = calculate_pixel_distance(left_hip_pixel,left_ankle_pixel)
            right_leg_lenght = calculate_pixel_distance(right_hip_pixel,right_ankle_pixel)
            #display distance 
            #cv2.putText(image, f'Left leg: {int(left_leg_lenght)}px',(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            #cv2.putText(image, f'Right leg: {int(right_leg_lenght)}px',(10,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
            
            #Classify posture besed on thresholds
            if left_leg_lenght < sitting_threshold and right_leg_lenght < sitting_threshold:
                posture =  "Sitting "
            else:
                 posture = "Standing"
                 
            cv2.putText(image,f"Posture: {posture}",(10,50),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)                
                    
        cv2.imshow('Pose Detection Basic', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()      