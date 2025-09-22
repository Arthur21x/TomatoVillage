# evaluate_efficientnet.py
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===== Caminhos =====
THIS_DIR = Path(__file__).resolve().parent            # .../EfficientNet
PROJ_DIR  = THIS_DIR.parent                           # raiz do projeto
DATASET_DIR = PROJ_DIR / "dataset"                    # dataset/{train,val,test}/{classe1,classe2}

# ===== Parâmetros =====
IMG_SIZE = (224, 224)
BATCH    = 32
EPOCHS   = 6
LR       = 1e-4

# ===== Geradores =====
# Usa o mesmo preprocess_input do EfficientNet. Augment apenas no treino.
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=35,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)
eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR / "train",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary",
    shuffle=True
)
val_gen = eval_datagen.flow_from_directory(
    DATASET_DIR / "val",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary",
    shuffle=False
)
test_gen = eval_datagen.flow_from_directory(
    DATASET_DIR / "test",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary",
    shuffle=False
)

# ===== Modelo (EfficientNetB0) =====
def build_efficientnet_b0_binary(img_size):
    inputs = Input(shape=img_size + (3,))
    base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
    # Fine-tuning leve: descongela últimas camadas
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers[-30:]:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs, name="efficientnet_b0_binary")
    return model

model = build_efficientnet_b0_binary(IMG_SIZE)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

# ===== Callbacks (sem salvar nada em disco) =====
callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4, verbose=1, min_lr=1e-6),
]

# ===== Treino =====
print("\n===== Treinando o modelo (EfficientNetB0) =====")
GPU = tf.config.list_physical_devices('GPU')
if GPU:
    try:
        with tf.device('/GPU:0'):
            history = model.fit(
                train_gen,
                epochs=EPOCHS,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1
            )
    except RuntimeError as e:
        print(e)
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
else:
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

# ===== Avaliação no TESTE (sem salvar nada) =====
print("\n===== Avaliando no conjunto de TESTE =====")
loss, acc = model.evaluate(test_gen, verbose=0)
print(f"Teste — loss: {loss:.4f} | acc: {acc:.4f}")

# ===== Predições e métricas detalhadas =====
y_true = test_gen.classes
class_names = list(test_gen.class_indices.keys())

y_prob = model.predict(test_gen, verbose=0).ravel()
y_pred = (y_prob > 0.5).astype("int32")

print("\nRelatório de Classificação:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cm, class_names, title="Matriz de Confusão — Teste"):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Verdadeiro",
        xlabel="Previsto",
        title=title,
    )
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

plot_confusion_matrix(cm, class_names)

# ===== Exibição de erros =====
errors_idx = np.where(y_true != y_pred)[0]
print(f"Foram encontrados {len(errors_idx)} erros no conjunto de teste.")

num_to_show = min(9, len(errors_idx))
if num_to_show > 0:
    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(errors_idx[:num_to_show]):
        img_path = test_gen.filepaths[idx]
        img = plt.imread(img_path)  # apenas visualização
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        true_label = class_names[int(y_true[idx])]
        pred_label = class_names[int(y_pred[idx])]
        plt.title(f"V: {true_label} | P: {pred_label}", color="red", fontsize=10)
        plt.axis("off")
    plt.suptitle("Exemplos de Erros — Teste", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
else:
    print("Não há erros para exibir.")