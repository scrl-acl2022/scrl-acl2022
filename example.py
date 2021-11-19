from scrl.model import load_model
from transformers import AutoTokenizer


def main():
    model_dir = "data/models/newsroom-L11/"
    device = "cpu"
    model = load_model(model_dir, device)
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    src = "This is a very long sentence that should be shortened."
    pred = model.predict([src], tokenizer, device)

    print(pred)


if __name__ == '__main__':
    main()
