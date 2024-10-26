from streamlit_webrtc import webrtc_streamer
import av
import cv2
import tensorflow as tf
import numpy as np
import imutils
import pickle
import os

# Path to models
model_path = 'liveness_model.h5'
mask_model_path = 'mask_detection_best.h5'
le_path = 'label_encoder.pickle'
encodings = 'encoded_faces.pickle'
detector_folder = 'face_detector'
confidence = 0.5

args = {'model': model_path, 'le': le_path, 'detector': detector_folder, 
        'encodings': encodings, 'confidence': confidence}

# Load the encoded faces and names
print('[INFO] loading encodings...')
with open(args['encodings'], 'rb') as file:
    encoded_data = pickle.loads(file.read())

# Load the face detector model
print('[INFO] loading face detector...')
proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# Load the liveness detector model and label encoder from disk
liveness_model = tf.keras.models.load_model(args['model'])
le = pickle.loads(open(args['le'], 'rb').read())

# Load the face mask detection model
mask_model = tf.keras.models.load_model(mask_model_path)


class VideoProcessor:        
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        # Resize the frame
        frm = imutils.resize(frm, width=800)

        # Blob for face detection
        (h, w) = frm.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frm, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # Detect faces
        detector_net.setInput(blob)
        detections = detector_net.forward()
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > args['confidence']:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Extract face ROI for liveness detection
                face = frm[startY:endY, startX:endX]
                if face.size == 0:
                    continue

                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = np.expand_dims(tf.keras.preprocessing.image.img_to_array(face), axis=0)

                # Liveness prediction
                preds = liveness_model.predict(face)[0]
                j = np.argmax(preds)
                label_name = le.classes_[j]
                label = f'{label_name}: {preds[j]:.4f}'

                # Mask detection (resize to 224x224 as expected by mask model)
                face_resized = cv2.resize(face[0], (224, 224))
                face_resized = np.expand_dims(face_resized, axis=0)

                mask_preds = mask_model.predict(face_resized)
                mask_label = 'Mask' if mask_preds[0][0] > 0.5 else 'No Mask'

                # Display the results
                cv2.putText(frm, f'{mask_label}', (startX, startY - 50),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0) if mask_label == 'Mask' else (0, 0, 255), 2)
                cv2.putText(frm, label, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                cv2.rectangle(frm, (startX, startY), (endX, endY), (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(frm, format='bgr24')

webrtc_streamer(key="key", video_processor_factory=VideoProcessor, rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }, sendback_audio=False, video_receiver_size=1)
