import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Objectron
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

# Initialize the Objectron model (you can change the object category)
objectron = mp_objectron.Objectron(static_image_mode=False,
                                 max_num_objects=5,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.99,
                                 model_name='Cup')

# Initialize webcam
cap = cv2.VideoCapture(0)

def draw_bounding_box(image, landmarks):
  h, w, c = image.shape
  x_coordinates = [landmark.x for landmark in landmarks]
  y_coordinates = [landmark.y for landmark in landmarks]
  
  x_min, x_max = int(min(x_coordinates) * w), int(max(x_coordinates) * w)
  y_min, y_max = int(min(y_coordinates) * h), int(max(y_coordinates) * h)
  
  cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

while cap.isOpened():
  success, image = cap.read()
  if not success:
      print("Ignoring empty camera frame.")
      continue

  # Convert the BGR image to RGB
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Process the image and detect objects
  results = objectron.process(image)

  # Convert the image color back so it can be displayed
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  if results.detected_objects:
      for detected_object in results.detected_objects:
          # Draw the box and axes
          mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
          mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)
          
          # Draw bounding box
          draw_bounding_box(image, detected_object.landmarks_2d.landmark)

  # Display the image
  cv2.imshow('MediaPipe Objectron', image)

  if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
      break

cap.release()
cv2.destroyAllWindows()

# Print the names of files created/modified during execution:
print("Created/Modified files during execution:")
print("No files were created or modified during this script's execution.")