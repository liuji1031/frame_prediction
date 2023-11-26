import tensorflow as tf
import tensorflow_addons as tfa
import keras
from loguru import logger

# def fpm_loss_fcn(x, y, y_pred, sample_weight=None):


# set up the encoder CNN
class EncoderNet(keras.Model):
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
            elif spec["type"]=="input":
                layers.append(tf.keras.layers.InputLayer(**spec["kwargs"]))

        self._seq = keras.Sequential(layers=layers)
        self.output_shapes = []

    def call(self,inputs,training=None,mask=None):
        """forward pass through all cnn layers

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        return self._seq(inputs)
    
    def compute_output_shape(self, input_shape):
        if len(input_shape)==3:
            input_shape = (None, *input_shape)
        shape = input_shape
        for l in self._seq.layers:
            shape = l.compute_output_shape(shape)
            self.output_shapes.append(shape)
        
        return shape
    
class DecoderNet(keras.Model):
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
            elif spec["type"]=="input":
                layers.append(tf.keras.layers.InputLayer(**spec["kwargs"]))
                
        self._seq = keras.Sequential(layers=layers)
        self.output_shapes = []
    
    def call(self, inputs, training=False, mask=None):
        return self._seq(inputs)
    
    def compute_output_shape(self, input_shape):
        if len(input_shape)==2:
            input_shape = (None, *input_shape)
        shape = input_shape
        for l in self._seq.layers:
            shape = l.compute_output_shape(shape)
            self.output_shapes.append(shape)
        
        return shape

class InteractionModule(keras.Model):
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
    
class FramePredictionModel(keras.Model):
    """connecting sub networks to produce the full frame prediction
    model

    Args:
        tf (_type_): _description_
    """

    def __init__(self,
                 encoder,
                 decoder,
                 interaction,
                 ):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.interaction = interaction
        self.r2 = tfa.metrics.r_square.RSquare(name='r2')

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
    
    def train_step(self,data):
        """performs one iteration of training

        Args:
            data (_type_): _description_
        """
        feature_in, frame_out = data
        # compute loss
        with tf.GradientTape() as tape:
            frame_out_pred = self(feature_in, training=True)
            loss_value = self.compute_loss(y=frame_out, y_pred=frame_out_pred)
        # apply gradient
        grad = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss_value)
            elif metric.name == "r2":
                # print('updating r2')
                metric.update_state(y_true=frame_out, y_pred=frame_out_pred)

        return {m.name: m.result() for m in self.metrics}