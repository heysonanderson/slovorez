# Slovorez

## SlovorezCXX Modules

### Prerequisites
- CMake 3.15+
- Compiler supporting C++11
- Python 3.8+ with development headers

---

### Build from Source

Install the required build dependencies

#### Ubuntu / Debian / Mint / Pop!\_OS

```bash
sudo apt install g++ cmake python3-dev
```

#### Fedora / Red Hat

```bash
sudo dnf install gcc-c++ cmake python3-devel
```

#### Arch / Manjaro / EndeavourOS

```bash
sudo pacman -S base-devel cmake python
```

#### macOS

Requires Homebrew

```bash
brew install cmake python@3.12
```

#### Windows

1. **C++ compiler** — Install Visual Studio with the *Desktop development with C++* workload, or MinGW-w64
2. **CMake** — Download from cmake.org
3. **Python** — Download from python.org. During installation, check **"Add Python to PATH"** and ensure **development headers** are included.

---

### Building

```bash
# Navigate to the project's folder
cd slovorez/

# Create build directory
mkdir build && cd build

# Configure and build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .

```

This should create shared library file in the root folder

---

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
