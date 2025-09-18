import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


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

img_height, img_width = 192, 192
batch_size = 16

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
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

for layer in base_model.layers:
    layer.trainable = False

# Modelo Final
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

os.makedirs("Model", exist_ok=True)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("Model/resnet_tomato.h5", monitor="val_loss", save_best_only=True)
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
