class FrameDataGenerator:
    def __init__(self, data_path, seed=None):
        """process the video data in data_path and 
        assign to train and validation set

        Args:
            data_path (_type_): _description_
        """

        self.data_path = data_path

        # find .avi and .txt files in data_path