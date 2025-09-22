# main_efficientnet_allinone.py
import os, math, hashlib, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# Config
# =========================
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

CAMINHO_RAIZ = Path(__file__).resolve().parent.parent   # ajuste se necessário
DATASET_DIR = os.path.join(CAMINHO_RAIZ, "dataset")     # dataset/{train,val,test}/{classe1,classe2}
IMG_SIZE = (224, 224)                                   # igual no treino e na avaliação
BATCH = 32
EPOCHS_WARMUP = 3
EPOCHS_FT = 6

# =========================
# Helpers
# =========================
def make_gen(split):
    dg = ImageDataGenerator(preprocessing_function=preprocess_input)
    gen = dg.flow_from_directory(
        os.path.join(DATASET_DIR, split),
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="binary",
        shuffle=(split == "train")  # embaralha só no treino
    )
    return gen

def check_leakage(train_gen, val_gen, test_gen, max_show=5):
    # interseção por NOME DE ARQUIVO
    def names(gen): return {Path(p).name for p in gen.filepaths}
    trn, val, tst = names(train_gen), names(val_gen), names(test_gen)
    n_tr_val = trn & val
    n_tr_tst = trn & tst
    n_val_tst = val & tst

    # interseção por HASH MD5 (custo: lê arquivos; ok para conjuntos pequenos/médios)
    def md5_set(gen):
        hs = set()
        for p in gen.filepaths:
            with open(p, "rb") as f: 
                hs.add(hashlib.md5(f.read()).hexdigest())
        return hs
    h_tr, h_val, h_tst = md5_set(train_gen), md5_set(val_gen), md5_set(test_gen)
    h_tr_val = h_tr & h_val
    h_tr_tst = h_tr & h_tst
    h_val_tst = h_val & h_tst

    # imprime resultados
    print("\n=== Checagem de Vazamento ===")
    print(f"Interseção (NOMES)  TR∩VAL: {len(n_tr_val)} | TR∩TEST: {len(n_tr_tst)} | VAL∩TEST: {len(n_val_tst)}")
    print(f"Interseção (HASH)   TR∩VAL: {len(h_tr_val)} | TR∩TEST: {len(h_tr_tst)} | VAL∩TEST: {len(h_val_tst)}")

    leaked = any([n_tr_val, n_tr_tst, n_val_tst, h_tr_val, h_tr_tst, h_val_tst])
    if leaked:
        print("\033[91m[ALERTA]\033[0m Vazamento detectado entre splits! Exemplos (nomes):")
        for title, s in [("TR∩VAL", n_tr_val), ("TR∩TEST", n_tr_tst), ("VAL∩TEST", n_val_tst)]:
            if s:
                print(f"  {title}: " + ", ".join(list(s)[:max_show]))
    else:
        print("\033[92m[OK]\033[0m Nenhum vazamento detectado por nome nem por hash.")

    return not leaked

def plot_confusion(cm, class_names, title):
    plt.figure(figsize=(6,5))
    plt.imshow(cm)
    plt.colorbar()
    plt.xticks(ticks=range(len(class_names)), labels=class_names)
    plt.yticks(ticks=range(len(class_names)), labels=class_names)
    plt.xlabel("Previsto"); plt.ylabel("Verdadeiro")
    plt.title(title)
    vmax = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            c = "white" if cm[i, j] > vmax/2 else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=c)
    plt.tight_layout()
    plt.show()

def show_misclassified(gen, y_true, y_pred, class_names, max_images=9, split_name="Validação"):
    errors = np.where(y_true != y_pred)[0]
    print(f"Foram encontrados {len(errors)} erros em {split_name}.")
    if len(errors) == 0:
        return
    n = min(max_images, len(errors))
    plt.figure(figsize=(12,12))
    for k, idx in enumerate(errors[:n]):
        img_path = gen.filepaths[idx]
        img = plt.imread(img_path)  # só para visualização
        plt.subplot(3,3,k+1)
        plt.imshow(img); plt.axis("off")
        t = class_names[int(y_true[idx])]
        p = class_names[int(y_pred[idx])]
        plt.title(f"V: {t} | P: {p}", color="red", fontsize=10)
    plt.suptitle(f"Exemplos de Erros — {split_name}", fontsize=14)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()

def evaluate_split(model, gen, split_label):
    class_names = list(gen.class_indices.keys())
    loss, acc = model.evaluate(gen, verbose=0)
    print(f"\n{split_label} — loss: {loss:.4f} | acc: {acc:.4f}")

    y_true = gen.classes
    y_prob = model.predict(gen, verbose=0).ravel()
    y_pred = (y_prob > 0.5).astype("int32")

    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, class_names, f"Matriz de Confusão — {split_label}")
    show_misclassified(gen, y_true, y_pred, class_names, split_name=split_label)

# =========================
# Dados
# =========================
print("Carregando dados...")
train_gen = make_gen("train")
val_gen   = make_gen("val")
test_gen  = make_gen("test")

print("train.class_indices:", train_gen.class_indices)
print("val.class_indices:  ", val_gen.class_indices)
assert train_gen.class_indices == val_gen.class_indices, "Classes/ordem diferentes entre train e val."

ok_no_leak = check_leakage(train_gen, val_gen, test_gen)
if not ok_no_leak:
    print("\033[91mRevise as pastas antes de confiar nas métricas.\033[0m")

steps_per_epoch  = math.ceil(train_gen.samples / BATCH)
validation_steps = math.ceil(val_gen.samples   / BATCH)

# =========================
# Modelo
# =========================
base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
base.trainable = False  # warmup

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.4)(x)
out = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(base.input, out)

# Warmup (cabeça)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
print("\n===== Treinando (warmup) =====")
model.fit(
    train_gen,
    epochs=EPOCHS_WARMUP,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    verbose=1
)

# Fine-tuning (parcial)
for layer in base.layers[:-20]:
    layer.trainable = False
for layer in base.layers[-20:]:
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
print("\n===== Treinando (fine-tuning) =====")
model.fit(
    train_gen,
    epochs=EPOCHS_FT,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    verbose=1
)

# =========================
# Avaliação final (val e test)
# =========================
print("\n===== Avaliação =====")
evaluate_split(model, val_gen,  "Validação")
evaluate_split(model, test_gen, "Teste")

# =========================
# Conclusão automática
# =========================
if ok_no_leak:
    print("\n\033[92m[CONCLUSÃO]\033[0m Sem vazamento detectado por nome/hash. "
          "Resultados consistentes para encerrar o dia — boa noite. 😴")
else:
    print("\n\033[93m[ATENÇÃO]\033[0m Houve indício de vazamento. "
          "Ajuste os diretórios antes de considerar o treinamento concluído.")