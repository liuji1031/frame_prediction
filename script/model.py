import tensorflow as tf
import tensorflow_addons as tfa
import keras
from loguru import logger


def get_encoder(input_dim, output_dim):
    layer_specs = [
                    {"type":"input","kwargs":{"input_shape":input_dim}}, #0
                    {"type":"conv2d","kwargs":{"filters": 128,  "kernel_size": 8, "strides":2,"activation":"relu"}}, #1
                    {"type":"conv2d","kwargs":{"filters": 128, "kernel_size": 6, "strides":2,"activation":"relu"}}, #2
                    {"type":"conv2d","kwargs":{"filters": 128, "kernel_size": 6, "strides":2,"activation":"relu"}}, #3
                    {"type":"conv2d","kwargs":{"filters": 128, "kernel_size": 4, "strides":2,"activation":"relu"}}, #4
                    {"type":"flatten"}, #5
                    {"type":"dense","kwargs":{"units":output_dim,"activation":"relu"}} #6
                 ]
    mdl = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_dim),
        tf.keras.layers.Conv2D(**layer_specs[1]["kwargs"]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(**layer_specs[2]["kwargs"]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(**layer_specs[3]["kwargs"]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(**layer_specs[4]["kwargs"]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(output_dim, activation='relu')
    ])

    return mdl

   
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


def get_interaction_module(
                encoder_dim=1024,
                action_dim=6,
                intermediate_dim=2048,
                output_dim=2048):
    
    img_enc_input = tf.keras.layers.Input(shape=(encoder_dim,))
    action_input = tf.keras.layers.Input(shape=(action_dim,))

    # the first fully connected layer that maps encoder to
    # interaction space (encoder_dim->intermediate_dim)
    # self.fc1 = tf.keras.layers.Dense(units=intermediate_dim,activation='linear')

    img_enc_output = tf.keras.Sequential([
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(intermediate_dim,activation=None),
    ])(img_enc_input)


    # the second fully connected layer that maps action to the
    # interaction space (action_dim->intermediate_dim)
    # self.fc2 = tf.keras.layers.Dense(units=intermediate_dim,activation='linear')

    action_output = tf.keras.Sequential(
        [   tf.keras.layers.Dense(units=256,activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=256,activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=intermediate_dim,activation='linear')
        ]
    )(action_input)

    # the third fully connected layer that mixes the interaction between action
    # and encoder
    # self.fc3 = tf.keras.layers.Dense(units=intermediate_dim)

    outputs = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(units=intermediate_dim,activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=output_dim,activation='relu'),
        ]
    )(img_enc_output*action_output)

    return tf.keras.Model(inputs=[img_enc_input,action_input],
                        outputs=outputs)

class custom_loss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        return tf.math.reduce_mean(tf.math.square(y_true-y_pred))
    
class FramePredictionModel(tf.keras.Model):
    """connecting sub networks to produce the full frame prediction
    model

    Args:
        tf (_type_): _description_
    """

    def __init__(self,
                 img_dim,
                 action_dim,
                 encoder,
                 interaction,
                 decoder
                 ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.interaction = interaction

        # img_input = tf.keras.layers.Input(shape=img_dim)
        # action_input = tf.keras.layers.Input(shape=(action_dim,))

        # enc_out = self.encoder(img_input)
        # int_out = self.interaction([enc_out, action_input])

        # outputs = self.decoder(int_out)

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
        # interaction = self.interaction(enc)

        # return decoder output
        return self.decoder(interaction)
    
    @tf.function
    def train_step(self, data):
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
                metric.update_state(y_true=frame_out, y_pred=frame_out_pred)

        return {m.name: m.result() for m in self.metrics}