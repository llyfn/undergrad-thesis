# undergrad-thesis

### Model Train
```bash
python src/main.py \
    --model_name roberta-base \
    --data data/sarcasm/train-balanced-sarcasm.csv \
    --output_dir ./out/contrastive \
    --batch_size 16 \
    --epochs 3 \
    --pretrain_epochs 2 \
    --lr 2e-5
```
