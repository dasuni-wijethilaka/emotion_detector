import cv2
import mediapipe as mp
import time

class FingerCounter:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Finger tip and pip (proximal interphalangeal) landmark IDs
        self.finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.finger_pips = [2, 6, 10, 14, 18]
        
    def count_fingers(self, hand_landmarks, handedness):
        """
        Count the number of raised fingers
        """
        if not hand_landmarks:
            return 0
        
        fingers_up = []
        
        # Get hand label (Left or Right)
        hand_label = handedness.classification[0].label
        
        # Thumb - different logic for left and right hand
        if hand_label == "Right":
            # For right hand: thumb is up if tip is to the right of pip
            if hand_landmarks.landmark[self.finger_tips[0]].x > hand_landmarks.landmark[self.finger_pips[0]].x:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        else:
            # For left hand: thumb is up if tip is to the left of pip
            if hand_landmarks.landmark[self.finger_tips[0]].x < hand_landmarks.landmark[self.finger_pips[0]].x:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        # Other four fingers - compare tip y-coordinate with pip y-coordinate
        for tip_id, pip_id in zip(self.finger_tips[1:], self.finger_pips[1:]):
            # If tip is above pip, finger is up (lower y value means higher position)
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        return sum(fingers_up)

def detect_palm():
    """
    Opens webcam for 5 seconds and detects the number of raised fingers
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access webcam")
        return None
    
    finger_counter = FingerCounter()
    
    print("Starting palm detection for 5 seconds...")
    print("Please show your palm to the camera!")
    
    start_time = time.time()
    finger_counts = []
    
    # Detection phase - 5 seconds
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= 5:
            break
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = finger_counter.hands.process(rgb_frame)
        
        # Display remaining time
        remaining = 5 - int(elapsed_time)
        cv2.putText(frame, f"Time remaining: {remaining}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw hand landmarks
                finger_counter.mp_draw.draw_landmarks(
                    frame, hand_landmarks, finger_counter.mp_hands.HAND_CONNECTIONS)
                
                # Count fingers
                num_fingers = finger_counter.count_fingers(hand_landmarks, handedness)
                finger_counts.append(num_fingers)
                
                # Display finger count
                cv2.putText(frame, f"Fingers: {num_fingers}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Palm Finger Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Calculate most common finger count
    if finger_counts:
        from collections import Counter
        count_frequency = Counter(finger_counts)
        most_common_count = count_frequency.most_common(1)[0][0]
        
        print(f"\n{'='*50}")
        print(f"Detection complete!")
        print(f"Most common finger count: {most_common_count}")
        print(f"All counts detected: {dict(count_frequency)}")
        print(f"{'='*50}")
    else:
        most_common_count = None
        print("No hand was detected during the scan")
    
    # Keep webcam open with final result
    print("\nWebcam is still active. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = finger_counter.hands.process(rgb_frame)
        
        # Display final result
        if most_common_count is not None:
            cv2.putText(frame, f"FINAL COUNT: {most_common_count} fingers", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(frame, "Press 'q' to exit", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw current hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_counter.mp_draw.draw_landmarks(
                    frame, hand_landmarks, finger_counter.mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow('Palm Finger Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    return most_common_count

if __name__ == "__main__":
    print("Palm Finger Detection System")
    print("="*50)
    finger_count = detect_palm()
    
    if finger_count is not None:
        print(f"\nFinal Result: You raised {finger_count} finger(s)")
    else:
        print("\nCould not detect hand. Please try again.")