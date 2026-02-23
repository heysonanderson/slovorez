# Slovorez

## SlovorezCXX Modules

### Prerequisites
- CMake 3.15+
- Compiler supporting C++11

### How to build

```bash
# Navigate to the project's folder
cd slovorez/

# Create build directory
mkdir build && cd build

# Configure and build
cmake .. && cmake --build .

# Run Lexer
./lexer
```

### Python installation

#### Navigate to project root

```bash
cd slovorez/
```

#### Activate venv (conda)

```bash
conda activate new_env
```

#### Install dependedncies
```bash
# CPU 
pip install .[tf]

# Tensorflow GPU
pip install .[tf-gpu]

# PyTorch GPU
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130 pip install .[torch]
```

#### Run demo
```bash
python -m src.main
```
