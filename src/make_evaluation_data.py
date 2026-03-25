from utils import load_ai_model4token_class

from peft import PeftModel


def main():
    ai_name = "BAAI/bge-m3"
    checkpoint = "data/checkpoints/single_class5/checkpoint-2"
    num_classes = 1110

    # lora_config = LoraConfig(
    #     r=64,
    #     lora_alpha=128,
    #     target_modules=["query", "key", "value", "dense"],
    #     modules_to_save=["classifier"],
    #     init_lora_weights="pissa_niter_10",
    # )

    ai_model = load_ai_model4token_class(ai_name, num_labels=num_classes)
    ai_model.float()

    ai_model = PeftModel.from_pretrained(ai_model, checkpoint)

    print(ai_model)


if __name__ == "__main__":
    main()
