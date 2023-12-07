import os
import pathlib
import random
from typing import Any, List
import tensorflow as tf
from loguru import logger
import cv2
import numpy as np
import time

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
    # values for normalizing action data
    action_mean = np.array([1.15083072e-01,
                            2.57701695e-01,
                            9.56139743e+00,
                            5.16164829e-03,
                            3.45287584e-03,
                            -1.62610730e-03,
                            8.38598115e+00])[np.newaxis,:]
    action_std = np.array([0.84561082,
                           2.28476457,
                           1.51305411,
                           0.09419529,
                           0.03367103,
                           0.05130835,
                           4.09652116])[np.newaxis]
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

        # use fold_n_frames to predict the next frame, default is 4
        self.fold_n_frames = config.get('fold_n_frames',4)
        self.frame_resize_shape = config.get('frame_resize_reshape',(192,256))

        # open all videos
        self.video_sources = [cv2.VideoCapture(str(path)) for path in self.file_list]
        # set a random offset for each video
        self.video_curr_frame = [] # the frame which the current video handle is on
        for src in self.video_sources:
            start_frame = np.random.randint(0,self.fold_n_frames+1)
            src.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.video_curr_frame.append(start_frame)

        self.video_length = [int(src.get(cv2.CAP_PROP_FRAME_COUNT)) for src in self.video_sources]
        
        self.n_video_finished = 0

        # read in all the arrays, first find all the filenames
        self.action_file_list = [path.parent / f"{path.stem}_merge.txt" for path in self.file_list]

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
            data = np.loadtxt(str(action_file))
            # data = (data-self.action_mean)/self.action_std

            # normalize according each file
            # data = (data-np.mean(data,axis=0,keepdims=True))/np.std(data,axis=0,keepdims=True)
            data = data/np.std(data,axis=0,keepdims=True)
            action_data.append(tf.convert_to_tensor(
                data,dtype=self.action_dtype))
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
        frame_std = 0.004 # rough estimate
        frame = (frame - 0.5)/frame_std # zero centered now

        # frame = tf.image.resize_with_pad(frame, *output_size)
        # permutate the array such that the shape is channel by height by width
        # frame = tf.transpose(frame,[2,0,1])
        return frame

    def unfinished_video_index(self):
        """check if finished going through all videos

        """
        index = []
        for vid_index, curr_frame in enumerate(self.video_curr_frame):
            if curr_frame + self.fold_n_frames < self.video_length[vid_index]:
                # print(f"vid{vid_index}, {curr_frame}")
                index.append(vid_index)
        return index
    
    def check_curr_frame(self, vid_no=None):
        """check for each video file whether the frames are
        exhausted. if so, revert to the beginning of the 
        file
        """
        # if will go out of video length, go back to the beginning
        # and randomly select the start
        if vid_no is None:
            for vid_index, curr_frame in enumerate(self.video_curr_frame):
                if curr_frame + self.fold_n_frames >= self.video_length[vid_index]:
                    start_frame = int(np.random.randint(0,self.fold_n_frames+1))
                    self.video_curr_frame[vid_index] = start_frame
                    self.video_sources[vid_index].set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            if self.video_curr_frame[vid_no] + self.fold_n_frames >= self.video_length[vid_no]:
                    start_frame = int(np.random.randint(0,self.fold_n_frames+1))
                    self.video_curr_frame[vid_no] = start_frame
                    self.video_sources[vid_no].set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def get_frames(self, vid_ind,start_frame_no=None):
        """grab frames from video specified by vid_ind

        Args:
            vid_ind (_type_): _description_
        """
        # set the start frame
        # curr_frame = self.video_curr_frame[vid_ind]
        src = self.video_sources[vid_ind]
        # src.set(cv2.CAP_PROP_POS_FRAMES, start_frame_no)

        # report status
        # logger.info((f"Loading {self.file_list[vid_ind].name}, "
        #         f"frame {start_frame_no} to {start_frame_no+self.fold_n_frames}"))

        # read the next several frames for frames_input
        frame_input, frame_output = None, None
        frames = None
        for _ in range(self.fold_n_frames+1):
            ret, frame = src.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if len(frame.shape)==2:
                frame = frame[:,:,np.newaxis]
            if not ret:
                return None,None
            
            if frames is None:
                frames = frame
            else:
                frames = np.concatenate((frames, frame),axis=2)

            # if ret:
            #     frame = self.format_frames(frame, self.frame_resize_shape)
            #     if frame_input is None:
            #         frame_input = frame # height by width by channel
            #     else:
            #         frame_input = tf.concat([frame_input, frame],axis=2)
            # else:
            #     return None, None
        # print(frames.shape)
        frames = tf.image.convert_image_dtype(frames, dtype=self.frame_dtype)
        frames = tf.image.resize(frames, self.frame_resize_shape)
        frames = (frames-0.5)/0.1
        # print(frames.shape)
        # the size of frame input is height x width x (channel x n folded frame)

        # read the next frame for frame_output
        
        # increment frame count
        self.video_curr_frame[vid_ind] += (self.fold_n_frames+1)

        frame_input = frames[:,:,:-1]
        # frame_output = (frames[:,:,-1]-frames[:,:,-2])[:,:,tf.newaxis]
        frame_output = (frames[:,:,-1])[:,:,tf.newaxis]
        # print(frame_input.shape, frame_output.shape)
        return frame_input, frame_output
    
    def get_action(self, vid_ind,start_frame_no):
        """get the action values

        Args:
            vid_ind (_type_): _description_
            start_frame_no (_type_): _description_
        """
        data = self.action_data[vid_ind][start_frame_no:start_frame_no+self.fold_n_frames,:]
        data = tf.gather(data,axis=1,indices=[0,1,5,6]) # keep only some directions of data, x,y acc/z ang v/fwd v
        data = tf.math.reduce_mean(data,axis=0)
        return data

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
            # self.check_curr_frame()

            # pick one
            vid_ind = int(np.random.randint(0,self.nvideos,size=None))

            # reset frame to beginning if necessary
            self.check_curr_frame(vid_no=vid_ind)

            # start_frame_no = self.sample_frame_no(vid_ind)
            # vid_indices = self.unfinished_video_index()
            # if len(vid_indices) == 0:
            #     print('Exhausted')
            #     return

            # randomly select one from vid_indices
            # vid_ind = vid_indices[int(np.random.randint(0,len(vid_indices),size=None))]
            start_frame_no = self.video_curr_frame[vid_ind]

            # logger.info(f'current vid index: {vid_ind}')

            # get frames and actions
            frame_input, frame_output = self.get_frames(vid_ind, start_frame_no)
            if frame_input is None:
                print('Exhausted')
                return
            actions = self.get_action(vid_ind, start_frame_no)
            # print(f'{vid_ind}, {start_frame_no}')

            yield (frame_input,actions), frame_output
        
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
    n_col = 7 # the txt files have 7 columns
    output_signature = ((tf.TensorSpec(shape = (None, None, 3*config["fold_n_frames"]),
                                    dtype = train_loader.frame_dtype),
                     tf.TensorSpec(shape = (n_col*config["fold_n_frames"],),
                                    dtype = train_loader.action_dtype)),
                    tf.TensorSpec(shape = (None, None, 3),
                                    dtype = train_loader.frame_dtype),
                    )
    
    train_ds = tf.data.Dataset.from_generator(train_loader,
                                          output_signature = output_signature)
    
    # for _ in range(20):
    #     frame_input_, frame_output_,actions_ = next(iter(train_ds))

    #     print(frame_input_.shape,frame_output_.shape, actions_.shape)
    #     print(actions_)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size = 20)
    train_ds = train_ds.batch(8)
    train_ds = train_ds.repeat()

    #for _ in range(5):
    prev_t = time.time()
    for data in train_ds:
        
        # frame_input_, frame_output_,actions_ = next(iter(train_ds))
        curr_t = time.time()
        print(curr_t-prev_t)
        prev_t = curr_t

    train_loader.cleanup()

if __name__ == "__main__":
    test()