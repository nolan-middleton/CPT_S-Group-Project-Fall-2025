#%% Setup

# Imports
import numpy as np
import os as os
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow as tf
import gc as gc

# VAE Classes
class Sampling(layers.Layer):
    '''
    Uses (mean, log_var) to sample z.
    
    Basically, takes in an array of mean values M and a variance (in log
    form) s, then returns a random vector drawn from norm(M[i], s) for all
    i.
    '''
    
    def call(self, inputs):
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name = "loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name = "reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name = "kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            mean, log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis = 1))
            total_loss = reconstruction_loss #+ kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply(grads, self.trainable_variables)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# Defs
def VAE_encoder_decoder(input_len, layer_lens):
    '''
    Creates the VAE encoder object.

    Parameters
    ----------
    input_len : int
        The dimension of the initial input layer.
    layer_lens : list
        A list of dimensions for the hidden layers. The last entry should be
        the latent dimension of the representation space.

    Returns
    -------
    keras.Model
        The encoder model.
    keras.Model
        The decoder model.
    '''
    # Setup
    input_shape = tuple([input_len])
    layer_shapes = [tuple([l]) for l in layer_lens]
    
    # Encoder
    encoder_inputs = keras.Input(shape = input_shape)
    x = layers.Dense(layer_lens[0], activation = "relu")(encoder_inputs)
    for i in range(1, len(layer_lens)):
        x = layers.Dense(layer_lens[i], activation = "relu")(x)
    
    mean = layers.Dense(layer_lens[-1], name = "mean")(x)
    log_var = layers.Dense(1, name = "log_var")(x)
    z = Sampling()([mean, log_var])
    encoder = keras.Model(encoder_inputs, [mean,log_var,z], name="encoder")
    
    # Decoder
    latent_inputs = keras.Input(shape = layer_shapes[-1])
    if (len(layer_lens) > 1):
        y = layers.Dense(layer_lens[-2], activation = "relu")(latent_inputs)
        for i in range(len(layer_lens) - 3, -1, -1):
            y = layers.Dense(layer_lens[i], activation = "relu")(y)
        decoder_output = layers.Dense(input_len)(y)
    else:
        decoder_output = layers.Dense(input_len)(latent_inputs)
    decoder = keras.Model(latent_inputs, decoder_output, name = "decoder")
    
    return (encoder, decoder)

# Variables
VAE_layers = [4096, 2048, 1024, 512]

#%% Main Loop
datasets = np.loadtxt("datasets.txt", dtype = str).tolist()

for dataset in datasets:
    print("> " + dataset + "...")
    
    #%%% Setup
    
    print(">> Setup...")
    with open(dataset + "/metadata.txt") as file:
        for line in file:
            if ("Classes: " in line):
                classes = line.replace("\n", "").split(": ")[1].split(", ")
                break
    labels = [c.split(".tsv")[0] for c in classes]
    if (not os.path.isdir(dataset + "/VAE")):
        os.mkdir(dataset + "/VAE")
    
    plain_X = np.loadtxt(dataset + "/1a/training_X.tsv", delimiter = "\t")
    plain_Y=np.loadtxt(dataset+"/1a/training_Y.tsv",delimiter="\t",dtype=int)
    
    print(">> Training VAE...")
    encoder, decoder = VAE_encoder_decoder(np.shape(plain_X)[1], VAE_layers)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer = keras.optimizers.Adam())
    vae.fit(plain_X, epochs = 20, batch_size = np.shape(plain_X)[0])
    gc.collect()
    
    embeddings = vae.encoder.predict(plain_X)
    np.savetxt(
        dataset + "/VAE/mean.tsv",
        embeddings[0],
        delimiter = "\t"
    )
    np.savetxt(
        dataset + "/VAE/log_var.tsv",
        embeddings[1],
        delimiter = "\t"
    )