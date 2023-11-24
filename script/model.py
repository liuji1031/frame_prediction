import tensorflow as tf
from loguru import logger

# set up the encoder CNN
class EncoderNet(tf.keras.Model):
    """the CNN for encoding a small image sequence
    e.g., n=4 consecutive frames

    Args:
        tf (_type_): _description_
    """
    def __init__(self, layer_specs = None):
        super().__init__()

        # layer_specs define the number of cnn layers,
        # the number of unit, and the kernel size
        layers = []
        for spec in layer_specs:
            # add the convolution layer
            if spec["type"]=="conv2d":
                layers.append(tf.keras.layers.Conv2D(**spec["kwargs"]))
            elif spec["type"]=="dense":
                layers.append(tf.keras.layers.Dense(**spec["kwargs"]))
            elif spec["type"]=="flatten":
                layers.append(tf.keras.layers.Flatten())
        self._layers = tf.keras.Sequential(layers=layers)

    def call(self,inputs,training=None,mask=None):
        """forward pass through all cnn layers

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        return self._layers(inputs)
    
    @property
    def output_shape(self):
        return self._layers.layers[-1].output_shape[1:]
    
class DecoderNet(tf.keras.Model):
    """the CNN for reconstructing the image from intermediate output

    Args:
        tf (_type_): _description_
    """
    def __init__(self,
                 layer_specs=None):
        super().__init__()

        # the first layer is a dense layer

        # iterate through all layer specs and define conv2d transpose 
        # layers
        layers = []
        for spec in layer_specs:
            if spec["type"]=="conv2dtr": # transposed conv2d layer
                layers.append(tf.keras.layers.Conv2DTranspose(**spec["kwargs"]))
            elif spec["type"]=="dense":
                layers.append(tf.keras.layers.Dense(**spec["kwargs"]))
            elif spec["type"]=="reshape":
                layers.append(tf.keras.layers.Reshape(**spec["kwargs"]))
        self._layers = tf.keras.Sequential(layers=layers)
    
    def call(self, inputs, training=False, mask=None):
        return self._layers(inputs)

    @property
    def output_shape(self):
        return self._layers.layers[-1].output_shape[1:]

class InteractionModule(tf.keras.Model):
    """this is the module for the encoded (embedded) features
    to interact with the action to output the decoder information

    Args:
        tf (_type_): _description_
    """
    def __init__(self,
                 encoder_dim=1024,
                 action_dim=6,
                 intermediate_dim=2048,
                 output_dim=2048):
        
        super().__init__()

        # the first fully connected layer that maps encoder to
        # interaction space (encoder_dim->intermediate_dim)
        self.fc1 = tf.keras.layers.Dense(units=intermediate_dim,activation='linear')

        # the second fully connected layer that maps action to the
        # interaction space (action_dim->intermediate_dim)
        self.fc2 = tf.keras.layers.Dense(units=intermediate_dim,activation='linear')

        # the third fully connected layer that mixes the interaction between action
        # and encoder
        self.fc3 = tf.keras.layers.Dense(units=intermediate_dim)

    def call(self,inputs,training=None,mask=None):
        """forward pass of the network

        Args:
            inputs (_type_): _description_
            training (_type_, optional): _description_. Defaults to None.
            mask (_type_, optional): _description_. Defaults to None.
        """
        # unpack input, input size should be batch size by D
        h_enc, action = inputs

        # feed h_enc through the first fc layer
        h_enc = self.fc1(h_enc)

        # feed action through the second fc layer
        action = self.fc2(action)

        # compute the interaction term (elementwise product)
        interaction = tf.math.multiply(h_enc,action)

        # feed interaction through the third layer and return result
        return self.fc3(interaction)
    
class FramePredictionModel(tf.keras.Model):
    """connecting sub networks to produce the full frame prediction
    model

    Args:
        tf (_type_): _description_
    """

    def __init__(self,
                 encoder,
                 decoder,
                 interaction,
                 learning_rate=1e-3
                 ):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.interaction = interaction

        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate,clipvalue=0.5)
        self.loss_fcn = tf.keras.losses.MeanSquaredError()

    def call(self, inputs, training=None, mask=None):
        """forward pass through the network

        Args:
            inputs (_type_): _description_
            training (_type_, optional): _description_. Defaults to None.
            mask (_type_, optional): _description_. Defaults to None.
        """
        # unpack inputs
        frame, action = inputs

        # encoder layer
        enc = self.encoder(frame)

        # interaction module
        interaction = self.interaction((enc, action))

        # return decoder output
        return self.decoder(interaction)
    
    def compute_loss(self, frame_in, frame_out, action, training=None):
        """compute loss

        Args:
            X (_type_): a tuple of (frame, action), where frame and action
            are each a batch of tensor of the appropriate dimension
            y (_type_): a batch of actual frame at the next time step
            training (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        frame_out_pred = self.call((frame_in, action))
        return self.loss_fcn(y_true=frame_out, y_pred=frame_out_pred)

    def get_batch(self, mode='training'):
        X = None
        y = None
        return X,y
    
    def compute_r2(self, X, y):
        """compute the R-squared value of the current model

        Args:
            X (_type_): _description_
            y (_type_): _description_
        """
        pass

    def step(self,mini_batch, verbose=False):
        """performs one iteration of training

        Args:
            mini_batch (_type_): _description_
        """

        # compute loss
        with tf.GradientTape() as tape:
            loss_value = self.compute_loss(*mini_batch)

            # apply gradient
            grad = tape.gradient(loss_value, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        
        if verbose:
            logger.info(f"loss: {loss_value.numpy()}")
        
    def evaluate(self):
        pass

    # def train(self,
    #           data=None,
    #           max_iteration=1e5,
    #           eval_iteration=100):
    #     """performs training

    #     Args:
    #         max_iteration (_type_, optional): _description_. Defaults to 1e5.
    #         eval_iteration (int, optional): _description_. Defaults to 100.
    #     """
    #     X_val = data["X_val"]
    #     y_val = data["y_val"]

    #     for i_iter in range(max_iteration):
            
    #         # fetch training data
    #         X,y = self.get_batch(mode="training")
            
    #         # compute loss
    #         with tf.GradientTape() as tape:
    #             loss_value = self.compute_loss(X,y,_)

    #         # apply gradient
    #         grad = tape.gradient(loss_value, self.trainable_variables)
    #         self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
            
    #         # for some interval report metrics on validation set
    #         if i_iter % eval_iteration == 0:
                
    #             self.compute_r2(X=X_val, y=y_val)
