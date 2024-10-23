import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

dataset_path = "../dataset"

def vectorize_images(images: np.ndarray) -> np.ndarray:
    vectors = []
    batch_size = 16

    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        batch_images = [Image.open(f"{dataset_path}/flickr30k_images/{img_path}") for img_path in batch_images]
        
        inputs = processor(images=batch_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            vectors.append(image_features.cpu().numpy())

    return np.vstack(vectors)


def vectorize_text(texts: np.ndarray) -> np.ndarray:
    vectors = []
    batch_size = 16

    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            vectors.append(text_features.cpu().numpy())

    return np.vstack(vectors)