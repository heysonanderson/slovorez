import logging
from slovorez import Slovorez

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

def main():
    model = Slovorez.from_pretrained("models/slovorez-test")
    logging.info("Model and components are ready. Processing.")
    model.process_file(
        file_path="text.txt"
    )

if __name__ == "__main__":
    main()