from tensorflow.keras.layers import RepeatVector, TimeDistributed, Input, LSTM, BatchNormalization, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_lstm_ae_model(
    lookback: int,
    n_features: int,
    lstm_units_1: int,
    latent_dim: int,
    learning_rate: float
) -> Model:
    """
    Build Simplified LSTM Autoencoder Model
    This architecture is less prone to overfitting and focuses on learning normal behavior
    """
    encoder_input = Input(shape=(lookback, n_features), name='encoder_input')
    
    # Encoder: Single LSTM sufficient for short sequences
    x = LSTM(lstm_units_1, return_sequences=False, name='encoder_lstm')(encoder_input)
    x = BatchNormalization(name='enc_bn')(x)
    
    # Dense bottleneck with reduced dimensionality
    latent = Dense(latent_dim, activation='relu', name='latent_space')(x)
    
    # Decoder
    x = RepeatVector(lookback, name='repeat_vector')(latent)
    
    # Decoder: Mirror encoder
    x = LSTM(lstm_units_1, return_sequences=True, name='decoder_lstm')(x)
    x = BatchNormalization(name='dec_bn')(x)
    
    # Output reconstruction
    decoder_output = TimeDistributed(
        Dense(n_features, activation='linear'),
        name='output'
    )(x)
    
    # Create model
    model = Model(encoder_input, decoder_output, name='LSTM_AE_FaultDetection')

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    # Display model architecture
    print("Simplified LSTM Autoencoder Architecture:")
    model.summary()

    print(f"\nAutoencoder Configuration:")
    print(f"  - Encoder: 1 LSTM layer ({lstm_units_1} units)")
    print(f"  - Latent Dimension: {latent_dim}")
    print(f"  - Decoder: 1 LSTM layer ({lstm_units_1} units)")
    print(f"  - Output: Reconstructed sequences of shape ({lookback}, {n_features})")
    print(f"  - Loss function: MSE (reconstruction loss)")
    print(f"  - Optimizer: Adam (lr={learning_rate})")
    print(f"  - Total Parameters: {model.count_params():,}")
    
    return model
