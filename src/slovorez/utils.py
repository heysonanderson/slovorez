from pathlib import Path
from typing import Union

# Путь к самой папке slovorez внутри site-packages или src
LIBRARY_ROOT = Path(__file__).resolve().parent.parent 
# Если data лежит рядом с src, нам нужно подняться выше:
PROJECT_ROOT = LIBRARY_ROOT.parent 

def resolve_path(path: Union[str, Path]) -> Path:
    p = Path(path)
    
    # 1. Если путь абсолютный — не трогаем его
    if p.is_absolute():
        return p
    
    # 2. Проверяем, существует ли файл относительно текущей рабочей директории (CWD)
    # Это то, что ожидает пользователь, передавая "my_data.txt"
    cwd_path = p.resolve()
    if cwd_path.exists():
        return cwd_path

    # 3. Если в CWD не нашли, проверяем внутри папок библиотеки (ваши словари)
    # Ищем в корне проекта (где лежит папка data)
    internal_path = (PROJECT_ROOT / p).resolve()
    if internal_path.exists():
        return internal_path

    # 4. Если нигде не нашли, возвращаем путь относительно CWD для стандартной ошибки
    return cwd_path

def file_exists(path: Union[str, Path]) -> bool:
    return resolve_path(path).is_file()

def dir_exists(path: Union[str, Path]) -> bool:
    return resolve_path(path).is_dir()