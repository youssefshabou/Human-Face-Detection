import cv2
import os

cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

def detect_faces(image, faceCascade, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE, confidence_threshold=0.2):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize, flags=flags)

    # Loop through the faces and draw a rectangle around each one
    for (x, y, w, h) in faces:
        confidence_score = 1.0  # Set the confidence score to 1.0 by default
        if confidence_threshold is not None:
            # Calculate the confidence score for the face
            confidence_score = (w * h) / (image.shape[0] * image.shape[1])
            if confidence_score < confidence_threshold:
                # If the confidence score is below the threshold, skip this face
                continue
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Write the confidence score on the image
        cv2.putText(image, f"Confidence: {confidence_score:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()

    # Detect faces in the frame
    try:
        frames = detect_faces(frames, faceCascade)
    except:
        pass

    # Display the resulting frame
    cv2.imshow('Video', frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
