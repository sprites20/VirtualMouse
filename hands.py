import cv2
import mediapipe as mp
import pandas as pd
import plotly.graph_objects as go
import time
import pyautogui

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

import pyvista as pv

import re

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5

def plot_landmarks(landmark_list, connections=None):
    if not landmark_list:
        return
    else:
        print(landmark_list)
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list):
        if (
            landmark.HasField("visibility")
            and landmark.visibility < _VISIBILITY_THRESHOLD
        ) or (
            landmark.HasField("presence") and landmark.presence < _PRESENCE_THRESHOLD
        ):
            continue
        plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
    if connections:
        out_cn = []
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx],
                    plotted_landmarks[end_idx],
                ]
                out_cn.append(
                    dict(
                        xs=[landmark_pair[0][0], landmark_pair[1][0]],
                        ys=[landmark_pair[0][1], landmark_pair[1][1]],
                        zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    )
                )
        cn2 = {"xs": [], "ys": [], "zs": []}
        for pair in out_cn:
            for k in pair.keys():
                cn2[k].append(pair[k][0])
                cn2[k].append(pair[k][1])
                cn2[k].append(None)

    df = pd.DataFrame(plotted_landmarks).T.rename(columns={0: "z", 1: "x", 2: "y"})
    df["lm"] = df.index.map(lambda s: mp_Hand.HandLandmark(s).name).values
    fig = (
        px.scatter_3d(df, x="z", y="x", z="y", hover_name="lm")
        .update_traces(marker={"color": "red"})
        .update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            scene={"camera": {"eye": {"x": 2.1, "y": 0, "z": 0}}},
        )
    )
    fig.add_traces(
        [
            go.Scatter3d(
                x=cn2["xs"],
                y=cn2["ys"],
                z=cn2["zs"],
                mode="lines",
                line={"color": "black", "width": 5},
                name="connections",
            )
        ]
    )

    return fig

# For static images:
"""
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
"""
# For webcam input:
cap = cv2.VideoCapture(0)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

some_arr = np.random.rand(21,3)
sc = ax.scatter(some_arr[:,0], some_arr[:,1], some_arr[:,2], color='green', s=0.5)
fig.show()

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
while True:
    indexfinger = 0
    with mp_hands.Hands(
      #model_complexity=0,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands:
      if cap.isOpened():
        success, image = cap.read()
        start = time.time()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
          #print("Starting")
          hand_coords_raw = {}
          hand_coord_processed = {}
          
          handlandmarks = results.multi_hand_landmarks
          multihandedness = results.multi_handedness
          hand_count = len(multihandedness)
          
          for i in range(hand_count):
            #print(multihandedness[i])
            #print(str(multihandedness[i].classification[0].index) + " : " + str(multihandedness[i].classification[0].label))
            #print(len(handlandmarks))
            hand_joint_count = len(handlandmarks[i].landmark)
            current_hand = str(multihandedness[i].classification[0].label)
            hand_coords_raw[current_hand] = {}
            hand_coord_processed[current_hand] = []
            
            for j in range(hand_joint_count):
                coord = handlandmarks[i].landmark[j]
                hand_coords_raw[current_hand][j] = [coord.x, coord.z, -coord.y + 1]
                hand_coord_processed[current_hand].append([coord.x, coord.z, -coord.y + 1])
                #print(j)
                #print(hand_coords_raw[current_hand][j])
            hand_coord_processed[current_hand] = np.array(hand_coord_processed[current_hand])
            
            
            #print(current_hand)
            at = None
      
          #print(hand_coord_processed[current_hand][8])
          indexfinger = hand_coord_processed[current_hand][8]
          #print(len(results.multi_hand_landmarks))
          #print(results.multi_hand_landmarks)
          #print(handlandmarks[8])
          #print(type(handlandmarks))
          
        
          """
          temp = vars(results)
          for item in temp:
            print( item , ' : ' , temp[item])
          """
          
          #print(len(results.multi_handedness))
          
          #print(results.multi_hand_landmarks[0].landmark[0].x)
          lenhand = len(handlandmarks)
          

          """
          indexesX = [i + 3 for i in range(lenhand) if handlandmarks.startswith("x: ", i)]
          indexesY = [i + 3 for i in range(lenhand) if handlandmarks.startswith("y: ", i)]
          indexesZ = [i + 3 for i in range(lenhand) if handlandmarks.startswith("z: ", i)]
          """
          #print(float(handlandmarks[indexesX[20]:indexesX[20]+20].rpartition('\n')[0]))
          #print(str(indexesX) + " " + str(indexesY) + " " +  str(indexesZ)) # <-- [6, 13, 19]
         # for item in results.multi_hand_landmarks:
            #print(i + item)
            #for item2 in item:
             #   print(item2)
          #plot_landmarks(results.multi_hand_landmarks,  mp_hands.HAND_CONNECTIONS)
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        """
        with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
            # Flip the image horizontally for a selfie-view display.
            results = pose.process(image)
            
            #print("AAAAAAAA")
            #print(results.pose_world_landmarks)
            
            # Draw the pose annotation on the image.
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        """
        end = time.time()
        # calculate the FPS for current frame detection
        fps = 1 / (end-start)
        # Show FPS
        cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        
        if cv2.waitKey(5) & 0xFF == 27:
          break
    """
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
      if cap.isOpened():
        #success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        # print(results)
        ###
        temp = vars(results)
        for item in temp:
          print( item , ' : ' , temp[item])
        ###
        #print(results.multi_face_landmarks)
        
        # Draw the face mesh annotations on the image.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
          #print((results.multi_face_landmarks[0].landmark[0]))
          
          landmark_count = len(results.multi_face_landmarks[0].landmark)
          face_coords = {}
          face_coords_processed = []
          for j in range(landmark_count):
                coord = results.multi_face_landmarks[0].landmark[j]
                face_coords[j] = [coord.x, coord.z, -coord.y + 1]
                face_coords_processed.append([coord.x, coord.z, -coord.y + 1])
                #print(j)
                #print(hand_coords_raw[current_hand][j])
          face_coords_processed = np.array(face_coords_processed)
            ###
          at = None
          try:
              at = face_coords_processed
              sc._offsets3d = (at[:,0], at[:,1], at[:,2])
              plt.draw()
          except:
            print("Error")
            ###
          print("FACE")

          for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        pose_landmarks_raw = []
        pose_landmarks_processed = []

        if results.pose_world_landmarks != None:
            print("Test")
            for i in results.pose_world_landmarks.landmark:
                pose_landmarks_raw = [i.x, i.z, -i.y + 1]
                pose_landmarks_processed.append([i.x, i.z, -i.y + 1])
                
                #print(j)
                #print(hand_coords_raw[current_hand][j])
            pose_landmarks_processed = np.array(pose_landmarks_processed)
            #print(pose_landmarks_processed)
        #print(results)
        temp = vars(results)
        for item in temp:
            #print( item , ' : ' , temp[item])
            a=0
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())            
        """
    shape = image.shape
    length = int(shape[1]-shape[1]/4) - int(shape[1]/4)
    width = length*0.5625/2
    
    x = pyautogui.size()[0] # getting the width of the screen
    y = pyautogui.size()[1] # getting the height of the screen
    
    start_point = (int(shape[1]/4), int((shape[0]/2) + length*0.5625/2))
    #print(image.shape)
    # Ending coordinate, here (125, 80)
    # represents the bottom right corner of rectangle
    end_point = (int(shape[1]-shape[1]/4), int((shape[0]/2) - length*0.5625/2))
    
    #print(indexfinger)
    
    boundsX = (start_point[0]/shape[1], end_point[0]/shape[1])
    boundsY = (start_point[1]/shape[0], end_point[1]/shape[0])
    #print(boundsX, boundsY)
    #print(int(x*(indexfinger[0]/boundsX[1])), int(y*(indexfinger[2]/boundsY[1])))
    
            
    try:
        print(indexfinger)
        if indexfinger[0] >= boundsX[0] and indexfinger[0] <= boundsX[1]:
            print(abs(x-int(x*indexfinger[0]/boundsX[0]))/2, abs(y-int(y*indexfinger[2]/boundsY[1])))
            print(boundsY)
            if indexfinger[2] <= boundsY[0] and indexfinger[2] >= boundsY[1]:
                #print(int(x*(indexfinger[0]/boundsX[1])), int(y*(indexfinger[2]/boundsY[1])))
                #print(y-int(y*indexfinger[2]/boundsY[1]))
                pyautogui.moveTo(int(abs(x-int(x*indexfinger[0]/boundsX[0]))/2), y-abs(y-int(y*indexfinger[2]/boundsY[1])))
    except:
        pass
    #print(length)
    #print(start_point)
    #print(end_point)
    # Black color in BGR
    color = (0, 0, 0)
       
    # Line thickness of -1 px
    # Thickness of -1 will fill the entire shape
    thickness = 5
       
    # Using cv2.rectangle() method
    # Draw a rectangle of black color of thickness -1 px
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    cv2.imshow('MediaPipe Hands', image)
    
    at = None
    try:
      if hand_count > 1:
        at = np.concatenate((hand_coord_processed["Left"], hand_coord_processed["Right"]), axis = 0)
      else:
        at = hand_coord_processed[current_hand]
      #print(at)
      
      #at = np.concatenate((at, face_coords_processed), axis = 0)
      #at = np.concatenate((at, pose_landmarks_processed), axis = 0)
      sc._offsets3d = (at[:,0], at[:,1], at[:,2])
      plt.draw()
    except:
        print("Error")
cap.release()