from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import TimeDistributed, Dense, GRU, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

def build_cnn_gru(num_classes=3, timesteps=5, img_size=224):
    # Base CNN (like DeepFake_Detection)
    base_cnn = InceptionV3(weights='imagenet', include_top=False,
                           input_shape=(img_size, img_size, 3))

    for layer in base_cnn.layers:
        layer.trainable = False  # start with frozen base

    # Wrap in TimeDistributed for sequence of frames
    video_input = Input(shape=(timesteps, img_size, img_size, 3))
    x = TimeDistributed(base_cnn)(video_input)
    x = TimeDistributed(GlobalAveragePooling2D())(x)  # (batch, timesteps, features)

    x = GRU(256, return_sequences=False)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=video_input, outputs=output)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
