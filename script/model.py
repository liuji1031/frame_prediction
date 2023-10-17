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


encoder_layer_specs = [
                        {"type":"conv2d","kwargs":{"filters": 64,  "kernel_size": 8, "stride":2,"activation":"relu"}},
                        {"type":"conv2d","kwargs":{"filters": 128, "kernel_size": 6, "stride":2,"activation":"relu"}},
                        {"type":"conv2d","kwargs":{"filters": 128, "kernel_size": 6, "stride":2,"activation":"relu"}},
                        {"type":"conv2d","kwargs":{"filters": 128, "kernel_size": 4, "stride":2,"activation":"relu"}},
                        {"type":"dense","kwargs":{"units":2048,"activation":"relu"}}
                    ]

decoder_layer_spces = [
                        {"type":"dense","kwargs":{"units":2048,"activation":"relu"}},
                        {"type":"conv2dtr","kwargs":{"filters": 128, "kernel_size": 4, "stride":2,"activation":"relu"}},
                        {"type":"conv2dtr","kwargs":{"filters": 128, "kernel_size": 6, "stride":2,"activation":"relu"}},
                        {"type":"conv2dtr","kwargs":{"filters": 128, "kernel_size": 6, "stride":2,"activation":"relu"}},
                        {"type":"conv2dtr","kwargs":{"filters":   3, "kernel_size": 8, "stride":2,"activation":"relu"}},
                    ]
