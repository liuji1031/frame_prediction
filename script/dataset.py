import os
from typing import Any, List
import tensorflow as tf
import random
from loguru import logger
import cv2
import pathlib
import numpy as np

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
            logger.info(f"Found video file: {file}")
        
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
        logger.info(f"Training: {len(self.train_set)} files")
        logger.info(f"Validation: {len(self.val_set)} files")
        logger.info(f"Test: {len(self.test_set)} files")
    
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
        file_list : List[pathlib.Path]
        self.file_list = file_list
        self.nvideos = len(file_list)
        self.frame_dtype = tf.float32
        self.action_dtype = tf.float32

        # open all videos
        self.video_sources = [cv2.VideoCapture(str(path)) for path in self.file_list]
        self.video_length = [int(src.get(cv2.CAP_PROP_FRAME_COUNT)) for src in self.video_sources]
        
        # # the frame which the current video handle is on
        self.video_curr_frame = [0 for _ in self.video_sources]

        self.n_video_finished = 0
        
        # use fold_n_frames to predict the next frame, default is 4
        self.fold_n_frames = config.get('fold_n_frames',4)
        self.frame_resize_shape = config.get('frame_resize_reshape',(192,256))

        # read in all the arrays, first find all the filenames
        self.action_file_list = [path.parent / f"{path.stem}.txt" for path in self.file_list]

        for action_file in self.action_file_list:
            logger.info(f"Found action file: {str(action_file)}")

        # read in all the action files, they are small in size
        self.action_data = self.read_action_files()
        self.check_vid_action_length()

    def check_vid_action_length(self):
        """check if each video has the same length
        as the action txt file
        """
        for i in range(self.nvideos):
            assert(self.video_length[i]==self.action_data[i].shape[0])
        logger.info('✅ All files match in length.')

    def read_action_files(self):
        """read in all the action txt files
        """
        action_data = []
        for action_file in self.action_file_list:
            action_data.append(tf.convert_to_tensor(
                np.loadtxt(str(action_file)),dtype=self.action_dtype))
        return action_data
    
    def format_frames(self, frame, output_size=(192,256)):
        """
            Pad and resize an image from a video.

            Args:
            frame: Image that needs to resized and padded. 
            output_size: Pixel size of the output frame image.

            Return:
            Formatted frame with padding of specified output size.
        """
        # convert from BGR to RGB first
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = tf.image.convert_image_dtype(frame, self.frame_dtype) # convert to [0,1)
        frame = tf.image.resize_with_pad(frame, *output_size)
        # permutate the array such that the shape is channel by height by width
        # frame = tf.transpose(frame,[2,0,1])
        return frame

    def unfinished_video_index(self):
        """check if finished going through all videos

        """
        index = []
        for vid_indeo, curr_frame in enumerate(self.video_curr_frame):
            if curr_frame + self.fold_n_frames < self.video_length[vid_indeo]:
                index.append(vid_indeo)
        return index
    
    def check_curr_frame(self):
        """check for each video file whether the frames are
        exhausted. if so, revert to the beginning of the 
        file
        """
        for vid_indeo, curr_frame in enumerate(self.video_curr_frame):
            if curr_frame + self.fold_n_frames >= self.video_length[vid_indeo]:
                self.video_curr_frame[vid_indeo] = 0

    def get_frames(self, vid_ind,start_frame_no):
        """grab frames from video specified by vid_ind

        Args:
            vid_ind (_type_): _description_
        """
        # set the start frame
        # curr_frame = self.video_curr_frame[vid_ind]
        src = self.video_sources[vid_ind]
        src.set(cv2.CAP_PROP_POS_FRAMES, start_frame_no)

        # report status
        logger.info((f"Loading {self.file_list[vid_ind].name}, "
                f"frame {start_frame_no} to {start_frame_no+self.fold_n_frames}"))

        # read the next several frames for frames_input
        frame_input = None
        for _ in range(self.fold_n_frames):
            ret, frame = src.read()
            if ret:
                frame = self.format_frames(frame, self.frame_resize_shape)
                if frame_input is None:
                    frame_input = frame # height by width by channel
                else:
                    frame_input = tf.concat([frame_input, frame],axis=2)
        # the size of frame input is height x width x (channel x n folded frame)

        # read the next frame for frame_output
        ret, frame = src.read()
        if ret:
            frame_output = self.format_frames(frame, self.frame_resize_shape)
        
        # increment frame count
        self.video_curr_frame[vid_ind] += 1

        return frame_input, frame_output
    
    def get_action(self, vid_ind,start_frame_no):
        """get the action values

        Args:
            vid_ind (_type_): _description_
            start_frame_no (_type_): _description_
        """
        data = self.action_data[vid_ind][start_frame_no:start_frame_no+self.fold_n_frames,:]
        return tf.reshape(data,[-1])

    def sample_frame_no(self, vid_no):
        """randomly selects a frame within the frame count of
        the video specified by the vid_no
        """
        nframes = self.video_length[vid_no]
        return np.random.randint(0, nframes-self.fold_n_frames-1)

    def __call__(self, *args: Any, **kwds: Any):
        """return frame data for training/validation/test

        """
        # for each iteration, randomly pick a video file that has frame left
        # and generate the folded frames as well as the frame to be predicted

        while True:
            # check for out of bound curr frame; reset if necessary
            self.check_curr_frame()

            # pick one 
            vid_ind = np.random.randint(0,self.nvideos,size=None)
            start_frame_no = self.sample_frame_no(vid_ind)

            # logger.info(f'current vid index: {vid_ind}')

            # get frames and actions
            frame_input, frame_output = self.get_frames(vid_ind,start_frame_no)
            actions = self.get_action(vid_ind, start_frame_no)

            yield frame_input, frame_output, actions
    
    def cleanup(self):
        """release all video handles
        """
        # # release all video sources when all finished
        for src in self.video_sources:
            src.release()
        logger.info("Released all video objects")

def test():
    """test data loader
    """
    data_manager = DataManager(
        data_path=r"/home/ji/Dropbox/Robotics/ENPM809K_Fundamentals_in_AI_and_DL/Data_Test",
        train_val_test_split=(1.0,0,0))
    
    config = {}
    config["fold_n_frames"] = 4

    train_loader = FrameDataGenerator(file_list=data_manager.get_training_files(),
                                      config=config)
    
    # for _ in range(5):
    #     frame_input_, frame_output_,actions_ = next(train_loader())

    #     print(actions_)
        # print(frame_input_.shape,frame_output_.shape)

        # plot the frames
        # for i in range(4):
        #     fi = tf.transpose(frame_input_[i*3:i*3+3,:,:],[1,2,0])
        #     fi = tf.image.convert_image_dtype(fi, tf.uint8).numpy()
        #     fi = cv2.cvtColor(fi, cv2.COLOR_RGB2BGR)
        #     cv2.imshow('test',fi)
        #     cv2.waitKey(0)

        # fo = tf.transpose(frame_output_,[1,2,0])
        # fo = tf.image.convert_image_dtype(fo, tf.uint8).numpy()
        # fo = cv2.cvtColor(fo, cv2.COLOR_RGB2BGR)

        # cv2.imshow('test',fo)
        # cv2.waitKey(0)

    # cv2.destroyAllWindows()

    # create tensorflow database from the generator
    output_signature = (tf.TensorSpec(shape = (None, None, 3*config["fold_n_frames"]),
                                      dtype = train_loader.frame_dtype),
                        tf.TensorSpec(shape = (None, None, 3),
                                      dtype = train_loader.frame_dtype),
                        tf.TensorSpec(shape = (6*config["fold_n_frames"],),
                                      dtype = train_loader.action_dtype))
    
    train_ds = tf.data.Dataset.from_generator(train_loader,
                                          output_signature = output_signature)
    
    # for _ in range(20):
    #     frame_input_, frame_output_,actions_ = next(iter(train_ds))

    #     print(frame_input_.shape,frame_output_.shape, actions_.shape)
    #     print(actions_)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size = AUTOTUNE)
    train_ds = train_ds.batch(8)

    for _ in range(5):
        frame_input_, frame_output_,actions_ = next(iter(train_ds))

        print(frame_input_.shape,frame_output_.shape, actions_.shape)

    train_loader.cleanup()

if __name__ == "__main__":
    test()