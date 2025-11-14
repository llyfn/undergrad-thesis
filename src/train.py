import torch
import json
import os
import copy
from datetime import datetime
from tqdm import tqdm

from evaluate import evaluate

def train(model, con_loss_fn, ce_loss_fn, train_loader, val_loader, test_loader, device, args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'config': vars(args),
        'pretrain_losses': [],
        'finetune_losses': [],
        'validation_metrics': [],
        'test_metrics': {}
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs * len(train_loader))
    for epoch in range(args.pretrain_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Pre-training Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            hidden = model(input_ids, attention_mask, is_pretraining=True)
            loss = con_loss_fn(hidden, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        results['pretrain_losses'].append(avg_loss)
        print(f"Pre-training Epoch {epoch+1}, Loss: {avg_loss}")

    best_macro_f1 = -1.0
    best_model_state = None
    patience = 2
    epochs_no_improve = 0

    optimizer_params = [
        {"params": model.classifier.parameters(), "lr": args.classifier_lr},
        {"params": model.model.parameters(), "lr": args.model_lr}
    ]
    optimizer = torch.optim.AdamW(optimizer_params)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.epochs * len(train_loader))
    for epoch in range(args.epochs):
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
            scheduler.step()
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

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"New best validation F1: {best_macro_f1:.4f}. Saving model.")
        else:
            epochs_no_improve += 1
            print(f"Validation F1 did not improve. Patience: {epochs_no_improve}/{patience}")
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No best model state was saved. Testing with the last model.")

    precision, recall, macro_f1 = evaluate(model, test_loader, device, desc="Testing")
    results['test_metrics'] = {
        'precision': precision,
        'recall': recall,
        'macro_f1': macro_f1
    }
    print(f"Test - Precision: {precision:.4f}, Recall: {recall:.4f}, Macro F1: {macro_f1:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"results_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")