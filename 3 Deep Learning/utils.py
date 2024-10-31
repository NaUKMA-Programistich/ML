from matplotlib import pyplot as plt
from transformers import GPT2TokenizerFast

import numpy as np
import torch

def visualize_samples(dataloader, count, text):
    batch = next(iter(dataloader))
    images, input_ids, _ = batch
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    plt.figure(figsize=(16, 12))
    for i in range(min(count, len(images))):
        image = images[i].permute(1, 2, 0).numpy()
        caption = tokenizer.decode(input_ids[i], skip_special_tokens=True)

        plt.subplot(count, 2, 2 * i + 1)
        plt.text(0.5, 0.5, caption, fontsize=12, ha='center', va='center', wrap=True)
        plt.axis('off')

        plt.subplot(count, 2, 2 * i + 2)
        plt.imshow(image)
        plt.axis('off')

    plt.suptitle(text)
    plt.tight_layout()
    plt.savefig(f'results/{text}.png')


def plot_metrics(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/losses.png')


def plot_sample_predictions(model, tokenizer, dataloader, num_samples=4):
    # idk
    pass