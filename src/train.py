import torch
import json
import os
from datetime import datetime
from tqdm import tqdm

from loss import ContrastiveLoss
from evaluate import evaluate

def train(model, train_loader, val_loader, test_loader, device, output_dir, epochs=5, pretrain_epochs=2, lr=2e-5):
    con_loss_fn = ContrastiveLoss()
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'pretrain_losses': [],
        'finetune_losses': [],
        'validation_metrics': [],
        'test_metrics': {}
    }

    for epoch in range(pretrain_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Pre-training Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            hidden = model(input_ids, attention_mask, is_pretraining=True)  # hidden 반환
            loss = con_loss_fn(hidden, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        results['pretrain_losses'].append(avg_loss)
        print(f"Pre-training Epoch {epoch+1}, Loss: {avg_loss}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = ce_loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        results['finetune_losses'].append(avg_loss)
        print(f"Fine-tuning Epoch {epoch+1}, Loss: {avg_loss}")

        precision, recall, macro_f1 = evaluate(model, val_loader, device, desc="Validating")
        results['validation_metrics'].append({
            'epoch': epoch + 1,
            'precision': precision,
            'recall': recall,
            'macro_f1': macro_f1
        })
        print(f"Validation - Precision: {precision:.4f}, Recall: {recall:.4f}, Macro F1: {macro_f1:.4f}")

    precision, recall, macro_f1 = evaluate(model, test_loader, device, desc="Testing")
    results['test_metrics'] = {
        'precision': precision,
        'recall': recall,
        'macro_f1': macro_f1
    }
    print(f"Test - Precision: {precision:.4f}, Recall: {recall:.4f}, Macro F1: {macro_f1:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")