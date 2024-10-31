from types import SimpleNamespace

import click
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
from transformers import GPT2TokenizerFast, get_linear_schedule_with_warmup

from model import CaptioningModel, FlickrDataset
from utils import plot_metrics, plot_sample_predictions, visualize_samples

model_config = SimpleNamespace(
    vocab_size=50_257,
    embed_dim=768,
    num_heads=12,
    seq_len=256,
    depth=6,
    attention_dropout=0.1,
    residual_dropout=0.1,
    mlp_ratio=4,
    mlp_dropout=0.1,
    emb_dropout=0.1,
)

def load_data(path: str) -> (np.array, np.array):
    data = pd.read_csv(f"{path}/labels.csv", delimiter='|', skipinitialspace=True)
    images = data['image_name'].tolist()
    descriptions = data['comment'].tolist()
    return images, descriptions


def collate_fn(batch):
    images, input_ids, attention_masks = zip(*batch)
    images = torch.stack(images)

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return images, input_ids, attention_masks


@click.command()
@click.option('--data_folder', type=str, default='../dataset')
@click.option('--bs', type=int, default=32)
@click.option('--device', type=str, default='cuda')
@click.option('--n_epochs', type=int, default=10)
@click.option('--lr', type=float, default=1e-4)
def main(data_folder, bs, device, n_epochs, lr):
    main_internal(data_folder, bs, device, n_epochs, lr)

def main_internal(data_folder, bs, device, n_epochs, lr):
    data = load_data(data_folder)
    dataset = FlickrDataset(data, f"{data_folder}/flickr30k_images")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Create datasets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False, collate_fn=collate_fn)

    # Visualize a few images and captions
    visualize_samples(train_dataloader, count=4, text="visualize_samples_train")
    visualize_samples(val_dataloader, count=4, text="visualize_samples_val")

    # Create tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model_config.eos_token_id = tokenizer.eos_token_id

    # Create model
    model = CaptioningModel(model_config)
    model.pretrained_layers_trainable(trainable=False)
    print(f'trainable parameters={sum([p.numel() for p in model.parameters() if p.requires_grad])}')
    model.to(device)

    # # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=len(train_dataloader) * n_epochs)

    # Define training loop
    train_losses = list()
    train_perplexities = list()

    # Define validation loop
    val_losses = list()
    val_perplexities = list()

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0

        for batch_idx, (images, input_ids, attention_masks) in enumerate(train_dataloader):
            images, input_ids, attention_masks = images.to(device), input_ids.to(device), attention_masks.to(device)
            labels = input_ids.clone()

            optimizer.zero_grad()
            loss = model(images, input_ids, labels=labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        train_perplexity = torch.exp(torch.tensor(avg_train_loss)).item()
        train_perplexities.append(train_perplexity)

        print(f'Epoch [{epoch + 1}/{n_epochs}], 'f'Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}')

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for images, input_ids, attention_masks in val_dataloader:
                images, input_ids, attention_masks = images.to(device), input_ids.to(device), attention_masks.to(device)

                labels = input_ids.clone()
                loss = model(images, input_ids, labels=labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        val_perplexities.append(val_perplexity)

        print(f'Epoch [{epoch + 1}/{n_epochs}], 'f'Val Loss: {avg_val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}')

    # Plot metrics
    plot_metrics(train_losses, val_losses)
    # Plot sample predictions
    plot_sample_predictions(model, tokenizer, val_dataloader)


if __name__ == '__main__':
    main_internal('../dataset', 4, 'mps', 5, 1e-4)