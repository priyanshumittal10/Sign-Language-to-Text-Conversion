import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
import mediapipe as mp

# Load the trained model
model_path = "D:\\Sign to Text Convertor Project\\Test\\orl_face_model.h5"
model = keras.models.load_model(model_path, compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Model input shape:", model.input_shape)

# Define the label mapping
label_mapping = {
    0: 'love', 1: 'thank_you', 2: 'hello', 3: 'goodbye', 4: 'yes',
    5: 'no', 6: 'please', 7: 'sorry', 8: 'help', 9: 'stop',
    10: 'go', 11: 'wait', 12: 'come', 13: 'good_morning', 14: 'good_night',
    15: 'congratulations', 16: 'birthday', 17: 'holiday', 18: 'vacation', 19: 'party',
    20: 'anniversary', 21: 'graduation', 22: 'success', 23: 'failure', 24: 'try_again',
    25: 'winner'
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Warm up the model
dummy_input = np.random.random((1,) + model.input_shape[1:])
_ = model.predict(dummy_input)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box coordinates
            h, w, _ = frame.shape
            x_max = y_max = 0
            x_min = w
            y_min = h
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
            
            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Crop and preprocess the hand region for prediction
            hand_frame = frame[y_min:y_max, x_min:x_max]
            if hand_frame.size != 0:  # Check if the cropped region is not empty
                resized_frame = cv2.resize(hand_frame, (model.input_shape[1], model.input_shape[2]))
                input_frame = resized_frame / 255.0
                input_frame = np.expand_dims(input_frame, axis=0)

                # Make prediction
                predictions = model.predict(input_frame, verbose=0)
                predicted_label = np.argmax(predictions)
                confidence = np.max(predictions)

                # Get the corresponding action label
                action_label = label_mapping[predicted_label]

                # Display the predicted label
                cv2.putText(frame, f"Action: {action_label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Sign Language Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()