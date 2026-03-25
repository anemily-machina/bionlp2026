from utils import load_ai_model4token_class

from peft import PeftModel


def main():
    ai_name = "BAAI/bge-m3"
    checkpoint = "data/checkpoints/single_class2/checkpoint-979"
    num_classes = 1110

    PeftModel.from_pretrained()

    ai_model = load_ai_model4token_class(ai_name, num_labels=num_classes)
    ai_model.float()

    ai_model = PeftModel.from_pretrained(ai_model, checkpoint)


if __name__ == "__main__":
    main()
