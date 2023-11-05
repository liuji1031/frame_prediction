import os
from typing import Any
import numpy as np
import tensorflow as tf
import random
from loguru import logger
import cv2
import pathlib

class DataManager:
    def __init__(self,
                 data_path,
                 seed=None,
                 train_val_test_split=(0.8,0.1,0.1)):
        """process the video data in data_path and 
        assign to train and validation set

        Args:
            data_path (_type_): _description_
        """

        self.data_path = pathlib.Path(data_path)
        self.file_list = []

        # find .avi and .txt files in data_path
        self.file_list = list(self.data_path.glob('*.avi'))
        for file in self.file_list:
            logger.info(f"Found file: {file}")
        
        # split into training, validation and test set
        if seed is not None:
            random.seed(seed)
        
        random.shuffle(self.file_list)
        nfile = len(self.file_list)

        # calculate number of files for each set
        nfile_train = int(nfile*train_val_test_split[0])
        nfile_val = int(nfile*sum(train_val_test_split[:2]))-nfile_train
        nfile_test = int(nfile*sum(train_val_test_split))-nfile_train-nfile_val
        assert(nfile_train+nfile_val+nfile_test<=nfile)

        # assign the file list according to file numbers
        self.train_set = self.file_list[:nfile_train]
        self.val_set = self.file_list[nfile_train:nfile_train+nfile_val]
        self.test_set = self.file_list[nfile_train+nfile_val:nfile_train+nfile_val+nfile_test]
        logger.info(f"training: {len(self.train_set)} files")
        logger.info(f"validation: {len(self.val_set)} files")
        logger.info(f"test: {len(self.test_set)} files")
    
    def get_training_files(self):
        """return a list of file names constituting the 
        training set

        Returns:
            _type_: _description_
        """
        return self.train_set
    
    def get_validation_files(self):
        """return a list of files belonging to the validation
        set

        Returns:
            _type_: _description_
        """
        return self.val_set
    
    def get_test_files(self):
        """return a list of files belonging to the test set

        Returns:
            _type_: _description_
        """
        return self.test_set
    

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class FrameDataGenerator:
    def __init__(self, file_list, mode="feedforward", config=None):
        """Initialize the frame generator."""
        self.file_list = file_list
        self.mode = mode
        self.config = config

    def __call__(self, *args: Any, **kwds: Any):
        """Generate frame data for training/validation/test."""
        # This method should be implemented to return the actual data
        pass

    def process_for_visual_odometry(self, video_path):
        """Process video frames for visual odometry."""
        cap = cv2.VideoCapture(video_path)
        ret, old_frame = cap.read()
        if not ret:
            print("Failed to read video")
            return None

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        # List to hold all the good points
        good_points_list = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Store points
            good_points_list.append((good_old, good_new))

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
                frame = cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)
            img = cv2.add(frame, mask)

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        cap.release()
        cv2.destroyAllWindows()

        # Return the list of good points for further processing
        return good_points_list

# Example usage:
if __name__ == "__main__":
    data_manager = DataManager(data_path=r"/home/ji/Dropbox/Robotics/ENPM809K_Fundamentals_in_AI_and_DL/Data")
    frame_data_generator = FrameDataGenerator(data_manager.get_training_files())
    if frame_data_generator.file_list:
        # Process the first video file for visual odometry as an example
        vo_data = frame_data_generator.process_for_visual_odometry(frame_data_generator.file_list[0])
    else:
        print("No video files found in the specified data path.")
