# TomatoVillage

Sistema de classificação de doenças em folhas de tomate utilizando redes neurais convolucionais (CNNs) com TensorFlow/Keras. O projeto inclui pipelines de treino, validação e teste, além de modelos pré-treinados prontos para inferência.

## Dataset (resumo)

- Estrutura: `dataset/{train,val,test}/{healthy,diseased}`.
- Uso:
  - train: treino (com augmentation)
  - val: validação/early stopping
  - test: avaliação final
- Dicas rápidas:
  - Balanceie healthy vs. diseased em cada split.
  - Imagens em RGB; formatos .jpg/.png.
  - O redimensionamento é feito pelo pipeline (ex.: 224x224).

Legenda: à esquerda, saudável; à direita, doente.
<p float="left">
  <img src="dataset/train/healthy/9bda0790-8cb2-4dce-8b8a-debdbd08a67d___GH_HL Leaf 388.jpg" alt="Folha saudável" width="45%">
  <img src="dataset/train/diseased/91f4f56a-b0c8-469f-88a7-47908efa5842___RS_Late.B 5437.jpg" alt="Folha doente" width="45%">
</p>

## Objetivo

- Detectar automaticamente se uma folha de tomate está saudável ou doente a partir de imagens.
- Oferecer uma base reproduzível para experimentos e comparações entre arquiteturas de visão computacional.

## Principais recursos

- Duas arquiteturas de referência:
  - EfficientNet-B0
  - ResNet (variação treinada para o domínio do tomate)
- Modelos salvos prontos para uso (.h5) e scripts de treino/teste.
- Estrutura de dados padronizada em pastas para train/val/test com classes `healthy` e `diseased`.

## Estrutura do projeto

- Os diretórios `EfficientNet` e `ResNet` contêm os pipelines específicos de cada arquitetura (treino e avaliação) e seus respectivos modelos salvos.

## Como executar

- Treinamento:
  - Escolha a arquitetura e execute o script de treino correspondente. Exemplos:
    - EfficientNet: executar `main.py` dentro de `EfficientNet/`
    - ResNet: executar `main.py` dentro de `ResNet/`
  - Certifique-se de que o dataset esteja no caminho padrão `dataset/` ou ajuste o caminho no script, se aplicável.

- Avaliação/Teste:
  - Utilize `test.py` dentro do diretório da arquitetura desejada para rodar inferência no conjunto `test/` e/ou em imagens individuais.
  - Os modelos `.h5` em `Model/` podem ser carregados diretamente para inferência.

Observação: parâmetros como taxa de aprendizado, épocas, tamanho de batch e augmentations podem estar configurados nos respectivos scripts da arquitetura.

## Métricas e resultados

- O projeto foi estruturado para reportar métricas de classificação como acurácia, precisão, recall e F1-score, além de matriz de confusão.
- Recomenda-se registrar resultados por arquitetura e conjunto (train/val/test) para comparações consistentes.

## Próximos passos

- Explorar EfficientNet variantes maiores (B1–B3) e técnicas de fine-tuning.
- Avaliar ensembles entre EfficientNet e ResNet.
- Adicionar explicabilidade (Grad-CAM) para interpretar regiões de atenção do modelo.
- Exportar modelos para formatos leves (SavedModel, TFLite) visando edge/IoT.

## Conclusões e Limitações do Treinamento

- Resultados (acurácia por classe no conjunto de teste):
  - ResNet: doentes 0.83 | saudáveis 0.92.
  - EfficientNet-B0: doentes 1.00 | saudáveis 1.00.
- Interpretação rápida:
  - ResNet apresenta pequena assimetria de desempenho entre classes, sugerindo possível confusão em casos limítrofes ou leve desbalanceamento.
  - EfficientNet com 1.00 em ambas as classes pode indicar excelente ajuste ao conjunto avaliado, mas exige verificação contra overfitting.
- Limitações e próximos passos:
  - Validar com conjunto externo (out-of-distribution) e/ou validação cruzada para confirmar generalização.
  - Aumentar diversidade do dataset (iluminação, ângulos, estágios da doença, diferentes câmeras).
  - Monitorar métricas por classe (precision/recall/F1) e matriz de confusão; calibrar limiar se necessário.
  - Incluir explicabilidade (ex.: Grad-CAM) para inspecionar regiões de atenção do modelo.
