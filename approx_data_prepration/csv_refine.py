import cv2
import csv
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

class joints_file_preparation:
    def __init__(self):
        pass
    
    def joints_labeling(self,joints_csv_path,save_json:bool=True
                        ,show:bool=True,
                        csv_pose_name_save:bool=True):
        
        # Define joint labels mapping
        joint_labels = {
            "Joint 0": "right_ankle",
            "Joint 1": "right_knee",
            "Joint 2": "right_hip",
            "Joint 3": "left_hip",
            "Joint 4": "left_knee",
            "Joint 5": "left_ankle",
            "Joint 6": "right_wrist",
            "Joint 7": "right_elbow",
            "Joint 8": "right_shoulder",
            "Joint 9": "left_shoulder",
            "Joint 10": "left_elbow",
            "Joint 11": "left_wrist",
            "Joint 12": "right_eye",
            "Joint 13": "left_eye",
            "Joint 14": "right_ear",
            "Joint 15": "left_ear",
            # Add more joint labels here as needed
        }

        # Read joint positions from the CSV file
        csv_file = joints_csv_path
        joint_positions = {}
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Get the header row
            for row in reader:
                joint = row[0]
                x, y, z = float(row[1]), float(row[2]), float(row[3])
                joint_label = joint_labels.get(joint, joint)  # Get the corresponding label or use the original joint name
                joint_positions[joint_label] = {"x": x, "y": y, "z": z}
        if save_json:
            # Save joint positions to a JSON file
            json_file = "joint_positions.json"
            with open(json_file, mode='w') as file:
                json.dump(joint_positions, file, indent=4)

        # Draw joints on the image and save joint positions with names in a new CSV file
        if csv_pose_name_save:
            csv_file_with_names = "pose_details_with_names.csv"
            with open(csv_file_with_names, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Joint", "X", "Y", "Z"])  # Write the header row
                for joint, position in joint_positions.items():
                    #x, y = int(position["x"] * img.shape[1]), int(position["y"] * img.shape[0])
                    #cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Draw a red circle at the joint position
                    writer.writerow([joint, position["x"], position["y"], position["z"]])
   
    def extract_coordinates(self,joint_dict):
            coordinates = {}
            joints=utils.DATA_CONFIG['joints']
            for joint in joints:
                # Iterate over the key-value pairs within each joint
                for (joint_key, joint_value) in joint.items():
                    coordinates[joint_value]=joint_dict[joint_key]
            return coordinates
