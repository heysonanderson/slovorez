import logging
from slovorez import Slovorez

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

def main():
    model = Slovorez.from_pretrained('models/slovorez-test')
    logging.info('Model and components are ready. Processing.')
    res = model.predict("Я сижу в своей комнате, в обиталище шума всей квартиры.")
    for r in res:
        print(r)

if __name__ == '__main__':
    main()