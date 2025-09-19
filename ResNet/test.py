import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Definindo caminhos
CAMINHO_RAIZ = Path(__file__).resolve().parent.parent
dataset_dir = os.path.join(CAMINHO_RAIZ, "dataset")
model_path = os.path.join(CAMINHO_RAIZ, "ResNet", "Model", "resnet_tomato.h5")  # Altere caso esteja usando outro nome de modelo

# Parâmetros de imagem
img_height, img_width = 192, 192
batch_size = 16

# Gerador para o conjunto de teste
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    os.path.join(dataset_dir, "test"),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False  # importante para bate com os rótulos
)

# Carrega o modelo salvo (.h5)
model = load_model(model_path)

# Avaliação no conjunto de teste
loss, acc = model.evaluate(test_generator)
print(f"✅ Acurácia no conjunto de teste: {acc * 100:.2f}%")

# Predições no conjunto de teste
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

# Classification Report
print("\n📊 Relatório de Classificação:")
print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.xlabel("Predição")
plt.ylabel("Verdadeiro")
plt.title("Matriz de Confusão")
plt.show()

# Mostrando alguns erros
errors_idx = np.where(y_true != y_pred)[0]
print(f"\nForam encontrados {len(errors_idx)} erros no teste.")

# Mostrar até 9 imagens mal classificadas
plt.figure(figsize=(12, 12))
for i, idx in enumerate(errors_idx[:9]):
    img_path = test_generator.filepaths[idx]
    img = plt.imread(img_path)
    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    true_label = list(test_generator.class_indices.keys())[y_true[idx]]
    pred_label = list(test_generator.class_indices.keys())[y_pred[idx]]
    plt.title(f"Verdadeiro: {true_label}\nPrevisto: {pred_label}", color="red")
    plt.axis("off")

plt.suptitle("Exemplos de erros do modelo", fontsize=16)
plt.show()