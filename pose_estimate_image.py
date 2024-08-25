import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Pose
    results = pose.process(image_rgb)
    
    # Draw pose landmarks on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image_rgb, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    
    return image_rgb, results

def main():
    # Specify the path to your image
    image_path = r'C:\Users\Aamir\OneDrive\Pictures\Family\AM0A4483.JPG'  # Update this path
    
    # Process the image
    processed_image, results = process_image(image_path)
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(processed_image)
    plt.title('Human Pose Estimation')
    plt.axis('off')
    plt.show()
    
    # Print landmark coordinates
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            print(f'Landmark {idx}: x={landmark.x}, y={landmark.y}, z={landmark.z}')
    else:
        print("No pose landmarks detected.")

if __name__ == "__main__":
    main()
