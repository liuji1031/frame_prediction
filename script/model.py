import tensorflow as tf

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
        for spec in layer_specs:
            # add the convolution layer
            if spec["type"]=="conv2d":
                self.layers.append(tf.keras.layers.Conv2D(**spec["kwargs"]))
            elif spec["type"]=="dense":
                self.layers.append(tf.keras.layers.Dense(**spec["kwargs"]))
    
    def call(self,inputs,training=None,mask=None):
        """forward pass through all cnn layers

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x
    
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
        for spec in layer_specs:
            if spec["type"]=="conv2dtr": # transposed conv2d layer
                self.layers.append(tf.keras.layers.Conv2DTranspose(**spec["kwargs"]))
            elif spec["type"]=="dense":
                self.layers.append(tf.keras.layers.Dense(**spec["kwargs"]))

class InteractionModule(tf.keras.Model):
    """this is the module for the encoded (embedded) features
    to interact with the action to output the decoder information

    Args:
        tf (_type_): _description_
    """
    def __init__(self,
                 encoder_dim=1024,
                 action_dim=6,
                 intermediate_dim=2048):
        
        super().__init__()

        # the first fully connected layer that maps encoder to
        # interaction space (1024->2048)
        self.fc1 = tf.keras.layers.Dense(units=intermediate_dim,activation='linear')

        # the second fully connected layer that maps action to the
        # interaction space (6->2048)
        self.fc2 = tf.keras.layers.Dense(units=intermediate_dim,activation='linear')

        # the third fully connected layer that mixes the interaction between action
        # and encoder
        self.fc2 = tf.keras.layers.Dense(units=intermediate_dim)

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

    default_encoder_layer_specs = [
                        {"type":"conv2d","kwargs":{"filters": 64,  "kernel_size": 8, "stride":2,"activation":"relu"}},
                        {"type":"conv2d","kwargs":{"filters": 128, "kernel_size": 6, "stride":2,"activation":"relu"}},
                        {"type":"conv2d","kwargs":{"filters": 128, "kernel_size": 6, "stride":2,"activation":"relu"}},
                        {"type":"conv2d","kwargs":{"filters": 128, "kernel_size": 4, "stride":2,"activation":"relu"}},
                        {"type":"dense","kwargs":{"units":1024,"activation":"relu"}}
                    ]
    
    default_decoder_layer_spces = [
                        {"type":"dense","kwargs":{"units":1024,"activation":"relu"}},
                        {"type":"conv2dtr","kwargs":{"filters": 128, "kernel_size": 4, "stride":2,"activation":"relu"}},
                        {"type":"conv2dtr","kwargs":{"filters": 128, "kernel_size": 6, "stride":2,"activation":"relu"}},
                        {"type":"conv2dtr","kwargs":{"filters": 128, "kernel_size": 6, "stride":2,"activation":"relu"}},
                        {"type":"conv2dtr","kwargs":{"filters":   3, "kernel_size": 8, "stride":2,"activation":"relu"}},
                    ]

    def __init__(self,
                 encoder_layer_specs=default_encoder_layer_specs,
                 decoder_layer_specs=default_decoder_layer_spces,
                 mode='feedforward',
                 learning_rate=1e-3
                 ):

        super().__init__()
        self.encoder = EncoderNet(layer_specs=encoder_layer_specs)
        self.decoder = DecoderNet(layer_specs=decoder_layer_specs)
        self.interaction = InteractionModule(intermediate_dim=2048)

        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate,clipvalue=0.5)
        self.loss_fcn = tf.keras.losses.mean_squared_error

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
    
    def compute_loss(self, X, y, training=None):
        """compute loss

        Args:
            X (_type_): a tuple of (frame, action), where frame and action
            are each a batch of tensor of the appropriate dimension
            y (_type_): a batch of actual frame at the next time step
            training (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        y_ = self.call(X)
        return self.loss_fcn(y_true=y, y_pred=y_)

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

    def train(self,
              data=None,
              max_iteration=1e5,
              eval_iteration=100):
        """performs training

        Args:
            max_iteration (_type_, optional): _description_. Defaults to 1e5.
            eval_iteration (int, optional): _description_. Defaults to 100.
        """
        X_val = data["X_val"]
        y_val = data["y_val"]

        for i_iter in range(max_iteration):
            
            # fetch training data
            X,y = self.get_batch(mode="training")
            
            # compute loss
            with tf.GradientTape() as tape:
                loss_value = self.compute_loss(X,y)

            # apply gradient
            grad = tape.gradient(loss_value, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
            
            # for some interval report metrics on validation set
            if i_iter % eval_iteration == 0:
                
                self.compute_r2(X=X_val, y=y_val)
