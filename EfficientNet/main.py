import tensorflow as tf
from pathlib import Path
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout


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


# Diretórios
CAMINHO_RAIZ = Path(__file__).resolve().parent.parent
dataset_dir = os.path.join(CAMINHO_RAIZ, "dataset")

train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")
test_dir = os.path.join(dataset_dir, "test")

img_height, img_width = 192, 192
batch_size = 32
epochs = 15

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,  # cuidado: só se fizer sentido no dataset
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Geradores de treino, teste e validação
train_generator = generator(train_datagen, train_dir, img_height, img_width, batch_size)
val_generator = generator(test_datagen, val_dir, img_height, img_width, batch_size)
test_generator = generator(test_datagen, test_dir, img_height, img_width, batch_size)


base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("Model/efficientNet_tomato.h5", monitor="val_loss", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1)
]

GPU = tf.config.list_physical_devices('GPU')
if GPU:
    try:
        with tf.device('/GPU:0'):
            history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=callbacks
            )
    except RuntimeError as e:
        print(e)
else:
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )

loss, acc = model.evaluate(test_generator)
print(f"\nAcurácia no conjunto de teste: {acc:.4f}")

