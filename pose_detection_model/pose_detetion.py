import cv2
import mediapipe as mp
import numpy as np
import csv
import warnings
warnings.filterwarnings("ignore")
import os 
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def joint_extraction(results):
    joint_dict = {}
    for i, landmark in enumerate(results.pose_landmarks.landmark):
        joint_dict[f'Joint {i}'] = {'X': landmark.x, 'Y': landmark.y, 'Z': landmark.z}
    return joint_dict

class pose_detection:
    def __init__(self):
        # Initialize mediapipe pose solution
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()

    def pose_estimation(self, image_path: str, pose_show=True, extracted_pose=True,
                        return_joints=True, save_csv=True):
        # Read the image
        img = cv2.imread(image_path)
        # Resize the image for better processing (optional)
        img = cv2.resize(img, (600, 400))

        # Perform pose detection
        self.results = self.pose.process(img)

        if self.results is None:
            print("Pose detection failed.")
            return None
        if self.results.pose_landmarks is None:
            print("Pose detection failed or no landmarks detected.")
            return None


        # Create a blank image with white background
        h, w, c = img.shape
        opimg = np.zeros([h, w, c], dtype=np.uint8)
        opimg.fill(255)

        # Draw the extracted pose on the blank image
        self.mp_draw.draw_landmarks(opimg, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                    self.mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                                    self.mp_draw.DrawingSpec((255, 0, 255), 2, 2))
        if pose_show:
            # Display the original image with pose estimation
            cv2.imshow("Pose Estimation", img)

        if extracted_pose:
            # Display the extracted pose on a blank image
            cv2.imshow("Extracted Pose", opimg)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if return_joints:
            # Extract joint locations
            joint_locations = []
            for landmark in self.results.pose_landmarks.landmark:
                joint_locations.append((landmark.x, landmark.y, landmark.z))
            if save_csv:
                # Save the joint positions and other details to a CSV file
                csv_file = "pose_details.csv"
                with open(csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Joint", "X", "Y", "Z"])  # Header row
                    for i, joint in enumerate(joint_locations):
                        writer.writerow([f"Joint {i}", joint[0], joint[1], joint[2]])
            return joint_extraction(self.results)
        else:
            return None
