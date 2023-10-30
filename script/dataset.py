import os
from typing import Any
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
    
class FrameDataGenerator:
    def __init__(self,
                 file_list,
                 mode="feedforward",
                 config=None):
        """return a new instance of the frame generator
        """
        self.file_list = file_list

    def __call__(self, *args: Any, **kwds: Any):
        """return frame data for training/validation/test

        """


if __name__ == "__main__":
    data_manager = DataManager(data_path=r"/home/ji/Dropbox/Robotics/ENPM809K_Fundamentals_in_AI_and_DL/Data")