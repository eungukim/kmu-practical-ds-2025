import tensorflow as tf
import numpy as np



def build_basic_ae(input_dim):
    tf.keras.backend.clear_session()

    # 입력 레이어
    input_layer = tf.keras.layers.Input(shape=(input_dim, ))

    # 인코더 레이어 정의
    encoder = tf.keras.layers.Dense(256, activation="relu")(input_layer)
    encoder = tf.keras.layers.Dense(128, activation="relu")(encoder)
    encoder = tf.keras.layers.Dense(64, activation="relu")(encoder)
    encoder = tf.keras.layers.Dense(32, activation="relu")(encoder)

    # 디코더 레이어 정의
    decoder = tf.keras.layers.Dense(64, activation='relu')(encoder)
    decoder = tf.keras.layers.Dense(128, activation='relu')(decoder)
    decoder = tf.keras.layers.Dense(256, activation='relu')(decoder)
    decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoder)

    # 모델 정의
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
    # 학습률 조정 및 클리핑 추가
    # 클리핑 : 너무 큰 값이 나오는 것을 방지하기 위해 사용
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    autoencoder.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    autoencoder.summary()

    return autoencoder

def build_dynamic_ae(input_dim, encode_layers=[32, 24, 16, 8], decode_layers=[16, 24, 32], learning_rate=0.001, loss='mse', metrics=['accuracy']):
    tf.keras.backend.clear_session()

    # 입력 레이어 정의
    input_layer = tf.keras.layers.Input(shape=(input_dim, ))
    encoder = input_layer
    
    # 인코더 레이어 구성
    # encode_layers 리스트의 각 값을 사용하여 Dense 레이어 생성
    for layer in encode_layers:
        encoder = tf.keras.layers.Dense(layer, activation="relu")(encoder)
        
    # 인코더의 출력을 디코더의 입력으로 사용
    decoder = encoder
    
    # 디코더 레이어 구성
    # decode_layers 리스트의 각 값을 사용하여 Dense 레이어 생성
    for layer in decode_layers:
        decoder = tf.keras.layers.Dense(layer, activation="relu")(decoder)
    
    # 출력 레이어 - 입력 차원과 동일한 크기로 복원
    # sigmoid 활성화 함수를 사용하여 0~1 사이 값으로 출력
    decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoder)
    # 모델 정의
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
    # 학습률 조정 및 클리핑 추가
    # 클리핑 : 너무 큰 값이 나오는 것을 방지하기 위해 사용
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    autoencoder.summary()

    return autoencoder
        
        
#적대적 오토인코더(AAE)
def build_aae(input_dim):
    tf.keras.backend.clear_session()

    # 입력 레이어
    input_layer = tf.keras.layers.Input(shape=(input_dim, ))
    
    # 인코더 네트워크
    encoder = tf.keras.layers.Dense(256, activation="relu")(input_layer)
    encoder = tf.keras.layers.Dense(128, activation="relu")(encoder)
    encoder = tf.keras.layers.Dense(64, activation="relu")(encoder)
    latent_code = tf.keras.layers.Dense(32, activation="linear")(encoder)
    
    # 디코더 네트워크
    decoder = tf.keras.layers.Dense(64, activation='relu')(latent_code)
    decoder = tf.keras.layers.Dense(128, activation='relu')(decoder)
    decoder = tf.keras.layers.Dense(256, activation='relu')(decoder)
    decoder_output = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoder)
    
    # 판별자 네트워크
    discriminator_input = tf.keras.layers.Input(shape=(32,))
    discriminator = tf.keras.layers.Dense(128, activation='relu')(discriminator_input)
    discriminator = tf.keras.layers.Dense(64, activation='relu')(discriminator)
    discriminator_output = tf.keras.layers.Dense(1, activation='sigmoid')(discriminator)
    
    # 모델 정의
    # 오토인코더 모델
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder_output)
    
    # 인코더 모델
    encoder_model = tf.keras.Model(inputs=input_layer, outputs=latent_code)
    
    # 디코더 모델
    decoder_input = tf.keras.layers.Input(shape=(32,))
    decoded = autoencoder.layers[-4](decoder_input)
    decoded = autoencoder.layers[-3](decoded)
    decoded = autoencoder.layers[-2](decoded)
    decoded = autoencoder.layers[-1](decoded)
    decoder_model = tf.keras.Model(inputs=decoder_input, outputs=decoded)
    
    # 판별자 모델
    discriminator_model = tf.keras.Model(inputs=discriminator_input, outputs=discriminator_output)
    
    # AAE 모델 (인코더 + 판별자)
    discriminator_model.trainable = False
    aae_input = tf.keras.layers.Input(shape=(input_dim,))
    aae_latent = encoder_model(aae_input)
    aae_output = discriminator_model(aae_latent)
    adversarial_model = tf.keras.Model(inputs=aae_input, outputs=aae_output)
    
    # 모델 컴파일
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    discriminator_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    adversarial_model.compile(optimizer='adam', loss='binary_crossentropy')
    
    return autoencoder, encoder_model, decoder_model, discriminator_model, adversarial_model

# =============================
# 개선된 오토인코더 및 분석 함수
# =============================

# 1. 더 깊은 층과 드롭아웃이 적용된 오토인코더

def build_deep_dropout_ae(input_dim, dropout_rate=0.3):
    """
    더 깊은 층과 드롭아웃이 적용된 오토인코더 모델을 생성합니다.
    """
    tf.keras.backend.clear_session()
    input_layer = tf.keras.layers.Input(shape=(input_dim, ))
    x = tf.keras.layers.Dense(512, activation="relu")(input_layer)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    encoded = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(encoded)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(x)
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()
    return autoencoder

# 2. 손실 함수 선택형 오토인코더

def build_ae_with_loss(input_dim, loss_fn='mae'):
    """
    손실 함수를 선택하여 오토인코더를 생성합니다. (mae, mse, huber 등)
    """
    tf.keras.backend.clear_session()
    input_layer = tf.keras.layers.Input(shape=(input_dim, ))
    x = tf.keras.layers.Dense(128, activation="relu")(input_layer)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    encoded = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(encoded)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(x)
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoded)
    if loss_fn == 'mae':
        loss = 'mae'
    elif loss_fn == 'huber':
        loss = tf.keras.losses.Huber()
    else:
        loss = 'mse'
    autoencoder.compile(optimizer='adam', loss=loss)
    autoencoder.summary()
    return autoencoder

# 3. 앙상블 오토인코더 클래스

class EnsembleAutoencoder:
    """
    여러 오토인코더를 앙상블하여 예측하는 클래스입니다.
    """
    def __init__(self, n_models, input_dim, build_fn=build_basic_ae):
        self.models = [build_fn(input_dim) for _ in range(n_models)]
        self.n_models = n_models
    def fit(self, x, epochs=50, batch_size=64, verbose=0):
        for model in self.models:
            model.fit(x, x, epochs=epochs, batch_size=batch_size, verbose=verbose)
    def predict(self, x):
        preds = np.array([model.predict(x) for model in self.models])
        return np.mean(preds, axis=0)
    def reconstruction_error(self, x):
        preds = self.predict(x)
        return np.mean(np.square(x - preds), axis=1)

# 4. 변분 오토인코더(VAE)

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(input_dim, latent_dim=2):
    # 인코더
    encoder_inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation='relu')(encoder_inputs)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # 디코더
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(128, activation='relu')(latent_inputs)
    decoder_outputs = tf.keras.layers.Dense(input_dim, activation='sigmoid')(x)
    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # VAE 모델
    class VAE(tf.keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder

        def train_step(self, data):
            if isinstance(data, tuple):
                x = data[0]
            else:
                x = data
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(x)
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(x, reconstruction))
                kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                total_loss = reconstruction_loss + kl_loss
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}

        def test_step(self, data):
            if isinstance(data, tuple):
                x = data[0]
            else:
                x = data
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(x, reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + kl_loss
            return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}

        def call(self, inputs):
            z_mean, z_log_var, z = self.encoder(inputs)
            return self.decoder(z)

    vae = VAE(encoder, decoder)
    vae.compile(optimizer='adam', loss=lambda y_true, y_pred: 0.0)  # dummy loss 추가
    vae.encoder.summary()
    vae.decoder.summary()
    return vae

# 5. 재구성 오차 기반 특성 중요도 분석

def feature_importance_by_reconstruction_error(x, model):
    """
    오토인코더의 재구성 오차를 기반으로 각 특성의 중요도를 계산합니다.
    """
    x_pred = model.predict(x)
    errors = np.abs(x - x_pred)
    importance = np.mean(errors, axis=0)
    importance = importance / np.sum(importance)
    return importance

# 