# undergrad-thesis

### Model Train
```bash
python src/main.py \
    --model_name roberta-base \
    --dataset figlang-reddit \
    --output_dir ./out/contrastive \
    --batch_size 64 \
    --epochs 3 \
    --pretrain_epochs 2 \
    --lr 5e-5
```
