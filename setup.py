import kagglehub
import os
import shutil
import random


def split_and_copy(files: list, class_name: str) -> None:
    """
    Divide uma lista de arquivos em treino, validação e teste e copia os arquivos para a pasta de dados do modelo.

    :param files: Lista de arquivos a serem divididos
    :type files: list
    :param class_name: nome da classe aos quais os arquivos pertencem
    :type class_name: str
    :return: None
    """
    random.shuffle(files)
    n_total = len(files)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:]
    }

    for split, imgs in splits.items():
        for img in imgs:
            dest = os.path.join(model_data, split, class_name, os.path.basename(img))
            shutil.copy(img, dest)


def apaga_pasta(caminho: str) -> None:
    """
    Apaga a pasta especificada, se ela existir.

    :param caminho: caminho para a pasta a ser apagada
    :type caminho: str
    :return: None
    """
    if os.path.exists(caminho):
        print(f"Apagando PlantVillage antigo: {caminho}")
        shutil.rmtree(caminho)


# 🔹 Diretório base onde o script está a rodar
base_path = os.getcwd()

local_dataset_path = os.path.join(base_path, "PlantVillage")

apaga_pasta(local_dataset_path)

# 🔹 Baixar dataset com kagglehub
print("📥 Baixando dataset do Kaggle...")
download_path = kagglehub.dataset_download("emmarex/plantdisease")

print(f"Copiando dataset para: {local_dataset_path}")
shutil.copytree(download_path, local_dataset_path)

# 🔹 Agora usamos a pasta local como origem
orig_dir = os.path.join(local_dataset_path, "PlantVillage")
model_data = os.path.join(base_path, "dataset")

# Se já existir dataset antigo, apaga antes de recriar
if os.path.exists(model_data):
    print(f"Apagando dataset antigo: {model_data}")
    shutil.rmtree(model_data)
os.makedirs(model_data, exist_ok=True)

# Criar estrutura de saída
for split in ["train", "val", "test"]:
    for cls in ["healthy", "diseased"]:
        os.makedirs(os.path.join(model_data, split, cls), exist_ok=True)

# 🔹 Remove pastas que não são tomate
for folder in os.listdir(orig_dir):
    folder_path = os.path.join(orig_dir, folder)
    if os.path.isdir(folder_path) and not folder.startswith("Tomato"):
        print(f"Removendo pasta não-Tomato: {folder_path}")
        shutil.rmtree(folder_path)

# 🔹 Divide saudável vs doente
healthy_dir = os.path.join(orig_dir, "Tomato_healthy")
diseased_dirs = [os.path.join(orig_dir, d) for d in os.listdir(orig_dir) if d != "Tomato_healthy"]

# Classe saudável
healthy_files = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir)]
split_and_copy(healthy_files, "healthy")

# Classe doente
diseased_files = []
for d in diseased_dirs:
    diseased_files.extend([os.path.join(d, f) for f in os.listdir(d)])
split_and_copy(diseased_files, "diseased")

apaga_pasta(local_dataset_path)

print("✅ Dataset processado e organizado em train/val/test dentro da pasta 'dataset'.")
