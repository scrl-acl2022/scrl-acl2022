# Sentence Compression with Reinforcement Learning

### Trained models, outputs, metrics

Coming soon

### Install `scrl` library
The library is required for training, using existing models and for evaluation.

**New environment**

We used Python 3.7 for this project, also works with Python 3.8 <br>
`python3.8 -m venv my_env` or with conda: `conda create -n my_env python=3.8` <br><br>
Activate the environment <br>
`source my_env/bin/activate` or with conda: `conda activate my_env`

**Install in development mode**

`pip install -r requirements.txt` <br>
`pip install -e .`

### Using trained model in Python

To run an existing model in Python, we need its model directory and we need to pick the correct pretrained model ID for the tokenizer, (usually `distilroberta-base`), corresponding to the original pretrained model that the sentence compressor was initialised with:
```python
from scrl.model import load_model
from transformers import AutoTokenizer

model_dir = "data/models/newsroom-L11/"
device = "cpu"
model = load_model(model_dir, device)
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

src = "This is a very long sentence that should be shortened."
summary = model.predict([src], tokenizer, device)
print(summary)
```

You can run this code with [example.py](example.py)

### Evaluation
Pick an evaluation dataset from `data/test-data` and a checkpoint of a model in `data/models`. <br>
Each trained model by default has a latest and a best checkpoint - "best" according to reward on holdout data.

```bash
python bin/evaluate.py \
  --checkpoint data/models/newsroom-L11/checkpoints/best_val_reward-7450 \
  --dataset data/test-data/google.jsonl \
  --device cpu
```
Optional settings, important for different test sets: <br>
`--verbose` to see predictions <br>
`--lower-src` to lowercase source text before compression <br>
`--lower-summary` to lowercase predicted and ground-truth summaries <br>
`--pretokenized` to white-space tokenize the ground-truth tokens <br>
`--max-chars 75` to truncate predictions to 75 tokens (for DUC2004 dataset) <br>


### Training a new model
Individual models and their training set, along with other settings, are defined in configuration files in the [config](config) folder. <br>

```
python bin/train.py \
  --config config/newsroom-test.json \
  --device cuda \
  --verbose
```

Training can simply be interrupted with `ctrl + C` and resumed with the same command. It will continue from the last saved checkpoint. <br>
Delete old model data and start from scratch by adding `--fresh`.
