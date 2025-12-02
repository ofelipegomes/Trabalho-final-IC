import torch
import clip
import cv2
import numpy as np
from PIL import Image

# Carrega o modelo CLIP 
# O modelo CLIP funciona aprendendo a relacionar imagens e textos.
# Ele foi treinado com milhões de pares do tipo imagem e descrição da imagem,
# que permite que ele aprenda não apenas a reconhecer formas,
# mas também o significado do que aparece nas imagens.

# Assim, quando recebe uma nova imagem, o modelo consegue gerar um vetor numérico que resume seu conteúdo visual de forma semântica,
# indicando o que há na imagem e não apenas suas cores ou bordas.
# Esses vetores compactos são chamados de embeddings e servem como uma representação rica,
# que facilita o trabalho dos classificadores tradicionais utilizados na etapa seguinte.


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def extract_deep_features(image, resize=(224, 224)):


    if image.shape[2] == 3:
        img = image[:, :, ::-1] 
    else:
        img = image

    img = preprocess(Image.fromarray(img.astype(np.uint8))).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.encode_image(img)

    feat = feat.cpu().numpy().astype("float32")
    return feat[0]
