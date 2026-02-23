import subprocess


class Lexer():
    """
    C++ Lexer API Wrapper
    """
    def run(self):
        cxxlexer = subprocess.run(
            ['./build/lexer'],
            capture_output=True,
            text=True
        )
        return cxxlexer.stdout