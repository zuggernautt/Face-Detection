import cv2
import dlib

# Load the pre-trained facial landmark detector from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/amit/Downloads/shape_predictor_68_face_landmarks.dat")

# Function to extract bounding box coordinates for eyes and mouth
def extract_bounding_boxes(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    bounding_boxes = []

    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(gray, face)

        # Extract the coordinates of the eyes and mouth
        left_eye_x = landmarks.part(36).x
        left_eye_y = landmarks.part(37).y
        right_eye_x = landmarks.part(45).x
        right_eye_y = landmarks.part(46).y
        mouth_left_x = landmarks.part(48).x
        mouth_left_y = landmarks.part(52).y
        mouth_right_x = landmarks.part(54).x
        mouth_right_y = landmarks.part(57).y

        # Calculate the coordinates for bounding boxes
        left_eye_bbox = (left_eye_x - 10, left_eye_y - 10, right_eye_x + 10, right_eye_y + 10)
        right_eye_bbox = (mouth_left_x - 10, mouth_left_y - 10, mouth_right_x + 10, mouth_right_y + 10)
        mouth_bbox = (mouth_left_x - 10, mouth_left_y - 10, mouth_right_x + 10, mouth_right_y + 10)

        bounding_boxes.append((left_eye_bbox, right_eye_bbox, mouth_bbox))

    return bounding_boxes

# Example usage
image_path = "00012.png"
bounding_boxes = extract_bounding_boxes(image_path)

# Print the coordinates of the bounding boxes
for i, (left_eye_bbox, right_eye_bbox, mouth_bbox) in enumerate(bounding_boxes):
    print(f"Bounding Box {i+1}:")
    print("Left Eye:", left_eye_bbox)
    print("Right Eye:", right_eye_bbox)
    print("Mouth:", mouth_bbox)
    print()
