import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers


def generator(datagen: ImageDataGenerator, diretorio: str, height: int, width: int,
              batch_size: int) -> ImageDataGenerator:
    """
    Gera um ImageDataGenerator a partir de um diretório de imagens.

    :param datagen: O ImageDataGenerator a ser usado.
    :type datagen: ImageDataGenerator
    :param diretorio: O diretório contendo as imagens.
    :type diretorio: str
    :param height: A altura das imagens.
    :type height: int
    :param width: A largura das imagens.
    :type width: int
    :param batch_size: O tamanho do batch de imagens.
    :type batch_size: int
    :return: Um ImageDataGenerator.
    :rtype: ImageDataGenerator
    """
    generador: ImageDataGenerator = datagen.flow_from_directory(
        os.path.join(dataset_dir, diretorio),
        target_size=(height, width),
        batch_size=batch_size,
        class_mode="binary"
    )
    return generador


# Diretório do Dataset
CAMINHO_RAIZ = Path(__file__).resolve().parent.parent
dataset_dir = os.path.join(CAMINHO_RAIZ, "dataset")

img_height, img_width = 224, 224  # Aumentado para melhor performance
batch_size = 32  # Aumentado para treino mais eficiente

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Geradores de treino, teste e validação
train_generator = generator(train_datagen, "train", img_height, img_width, batch_size)
val_generator = generator(test_datagen, "val", img_height, img_width, batch_size)
test_generator = generator(test_datagen, "test", img_height, img_width, batch_size)

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

# Fine tuning em mais camadas
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Modelo Final com mais camadas e regularização
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.0005)),
    Dropout(0.4),
    Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.0005)),
    Dropout(0.3),
    Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.0005))
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

os.makedirs("Model", exist_ok=True)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ModelCheckpoint("Model/resnet_tomato.h5", monitor="val_loss", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4, verbose=1, min_lr=1e-6)
]
GPU = tf.config.list_physical_devices('GPU')
if GPU:
    try:
        with tf.device('/GPU:0'):
            history = model.fit(
                train_generator,
                epochs=10,
                validation_data=val_generator,
                callbacks=callbacks
            )
    except RuntimeError as e:
        print(e)
else:
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=callbacks
    )

loss, acc = model.evaluate(test_generator)
print(f"✅ Acurácia no conjunto de teste: {acc * 100:.2f}%")
