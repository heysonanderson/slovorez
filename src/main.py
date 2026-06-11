from slovorez import Slovorez


def main():
    model = Slovorez.from_pretrained('models/slovorez-test', device='cpu')
    res = model.predict("Я сижу в своей комнате, в обиталище шума всей квартиры.")
    for r in res:
        print(r)

if __name__ == '__main__':
    main()