import argparse
import torch
from torch.utils.data import DataLoader
import os
import json

from types import SimpleNamespace
from dataset import load_dataset
from loss import ContrastiveLoss
from model import SarcasmModel
from train import train

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset, val_dataset, test_dataset = load_dataset(args.dataset, args.model_name, val_ratio=0.1)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = SarcasmModel(args.model_name, dropout=args.dropout).to(device)
    con_loss_fn = ContrastiveLoss(temperature=args.temperature)
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    train(model, con_loss_fn, ce_loss_fn, train_loader, val_loader, test_loader, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sarcasm Detection with Contrastive Learning")
    parser.add_argument('--config', type=str, default='config.json', help='Path to the JSON configuration file')
    with open(parser.parse_args().config, 'r') as f:
        config = json.load(f)
    main(SimpleNamespace(**config))