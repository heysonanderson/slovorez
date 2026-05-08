import logging
from slovorez.slovorez import Slovorez

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

def main():
    # path configuration
    CONFIG_PATH = "data/ml/models/onnx/slovorez-test.json"
    BASE_DICT_PATH = "data/dictionaries/static_dictionary/tikhonov-morphemes-pos.json"
    TEXT_PATH = "large.txt"
    OUTPUT_PATH = "data/dictionaries/model_outputs/predictions-raw-test.jsonl"

    model = Slovorez.from_pretrained(
        config_path=CONFIG_PATH,
        base_dict_path=BASE_DICT_PATH,
        output_path=OUTPUT_PATH,
        device="auto" # "cuda" or "cpu"
    )

    logging.info("Model and components are ready. Processing")

    model.process_file(
        file_path=TEXT_PATH,
        batch_size=65536,    # File batch size
        model_batch=2048,    # Model batch size
        max_workers=8        # CPU workers max count
    )

if __name__ == "__main__":
    main()