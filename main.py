import subprocess

cxxlexer = subprocess.run(
    ['./build/lexer'],
    capture_output=True,
    text=True
)
print(cxxlexer.stdout);
