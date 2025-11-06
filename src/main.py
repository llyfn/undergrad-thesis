import argparse
import torch
from torch.utils.data import DataLoader
import os

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

    train(model, con_loss_fn, ce_loss_fn, train_loader, val_loader, test_loader, device, args.output_dir, args.epochs,
          args.pretrain_epochs, args.lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sarcasm Detection with Contrastive Learning")
    parser.add_argument('--model_name', type=str, default='roberta-base', help='Base model name (e.g. bert-base-cased, roberta-base)')
    parser.add_argument('--dataset', type=str, default='figlang-reddit', choices=['figlang-reddit', 'figlang-twitter', 'sarc-reddit'], help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='./out', help='Directory to save model checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Fine-tuning epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=2, help='Pre-training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Classifier Learning rate')
    parser.add_argument('--model_lr', type=float, default=2e-5, help='Model Learning rate')
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate for the model.")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for the contrastive loss.")
    args = parser.parse_args()
    main(args)