import tensorflow as tf
from tensorflow.keras.layers import RepeatVector, TimeDistributed, Input, LSTM, BatchNormalization, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the latent representation."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_lstm_vae_model(
    lookback: int,
    n_features: int,
    lstm_units_1: int,
    latent_dim: int,
    learning_rate: float
) -> Model:
    """
    Build Simplified LSTM Variational Autoencoder Model (VAE)
    This architecture converts the continuous latent space into a probability distribution.
    """
    encoder_input = Input(shape=(lookback, n_features), name='encoder_input')
    
    # Encoder: Single LSTM sufficient for short sequences
    x = LSTM(lstm_units_1, return_sequences=False, name='encoder_lstm')(encoder_input)
    x = BatchNormalization(name='enc_bn')(x)
    
    # VAE: Z_mean and Z_log_var bottleneck
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    
    # VAE: Sampling layer
    z = Sampling(name='sampling')([z_mean, z_log_var])
    
    # Decoder
    x = RepeatVector(lookback, name='repeat_vector')(z)
    
    # Decoder: Mirror encoder
    x = LSTM(lstm_units_1, return_sequences=True, name='decoder_lstm')(x)
    x = BatchNormalization(name='dec_bn')(x)
    
    # Output reconstruction
    decoder_output = TimeDistributed(
        Dense(n_features, activation='linear'),
        name='output'
    )(x)
    
    # Create model
    model = Model(encoder_input, decoder_output, name='LSTM_VAE_FaultDetection')

    # KL Divergence Loss Calculation
    kl_loss = -0.5 * tf.reduce_mean(
        z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
    )
    model.add_loss(kl_loss)

    # Compile model (Reconstruction Loss + Added KL Loss)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    # Display model architecture
    print("Simplified LSTM Variational Autoencoder (VAE) Architecture:")
    model.summary()

    print(f"\nVariational Autoencoder Configuration:")
    print(f"  - Encoder: 1 LSTM layer ({lstm_units_1} units)")
    print(f"  - Latent Dimension: {latent_dim} (represented by mean and log_var)")
    print(f"  - Decoder: 1 LSTM layer ({lstm_units_1} units)")
    print(f"  - Output: Reconstructed sequences of shape ({lookback}, {n_features})")
    print(f"  - Loss function: MSE (reconstruction loss) + KL Divergence")
    print(f"  - Optimizer: Adam (lr={learning_rate})")
    print(f"  - Total Parameters: {model.count_params():,}")
    
    return model
