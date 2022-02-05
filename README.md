# HTR
Codebase for training various HTR models.

Current models include:
- FPHTR
- Show Attend and Read

## How to install
```shell
pip install -e .
```

## Examples
### FPHTR
```shell
python htr/main.py --data_dir /path/to/IAM \
                   --model fphtr \
                   --data_format word \
                   --max_epochs 3 \
                   --precision 16 \
                   --use_aachen_splits \
```

### SAR
```shell
python htr/main.py --data_dir /path/to/IAM \
                   --model sar \
                   --max_epochs 3 \
                   --precision 16 \
                   --use_aachen_splits \
```
