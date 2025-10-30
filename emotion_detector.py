import cv2
from deepface import DeepFace
import time
from collections import Counter

def detect_emotion():
    """
    Captures video from webcam for 5 seconds and detects emotions.
    Returns the most common emotion detected during that period.
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access webcam")
        return None
    
    print("Starting emotion detection for 5 seconds...")
    print("Please look at the camera!")
    
    start_time = time.time()
    emotions_detected = []
    
    while True:
        # Check if 5 seconds have passed
        elapsed_time = time.time() - start_time
        if elapsed_time >= 5:
            break
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Display remaining time on frame
        remaining = 5 - int(elapsed_time)
        cv2.putText(frame, f"Time remaining: {remaining}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        try:
            # Analyze emotion using DeepFace
            result = DeepFace.analyze(frame, actions=['emotion'], 
                                     enforce_detection=False, silent=True)
            
            # Handle both single face and multiple faces results
            if isinstance(result, list):
                result = result[0]
            
            dominant_emotion = result['dominant_emotion']
            emotions_detected.append(dominant_emotion)
            
            # Display detected emotion on frame
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
        except Exception as e:
            # If no face detected or error occurs
            cv2.putText(frame, "No face detected", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow('Emotion Detection', frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Determine the most common emotion
    if emotions_detected:
        emotion_counts = Counter(emotions_detected)
        most_common_emotion = emotion_counts.most_common(1)[0][0]
        
        print(f"\n{'='*50}")
        print(f"Detection complete!")
        print(f"Most common emotion detected: {most_common_emotion.upper()}")
        print(f"All emotions detected: {dict(emotion_counts)}")
        print(f"{'='*50}")
    else:
        most_common_emotion = None
        print("No emotions were detected during the scan")
    
    # Keep showing the webcam with final result
    print("\nWebcam is still active. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display final result on frame
        if most_common_emotion:
            cv2.putText(frame, f"FINAL EMOTION: {most_common_emotion.upper()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to exit", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No emotion detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Press 'q' to exit", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    return most_common_emotion

if __name__ == "__main__":
    print("Face Emotion Detection System")
    print("="*50)
    detected_emotion = detect_emotion()
    
    if detected_emotion:
        print(f"\nFinal Result: Your emotion is '{detected_emotion}'")
    else:
        print("\nCould not determine emotion. Please try again.")