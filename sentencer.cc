#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "text_lexer.h"

constexpr size_t MAX_WORDS_SIZE = 65536;

namespace py = pybind11;

class FromTextSentencer {
private:
    size_t len = 0;
    size_t pos = 0;
    std::vector<Token> tokens_batch;
    char* raw_text = nullptr;

public:
    FromTextSentencer(const char* str, size_t str_len) : len(str_len), pos(0)
    {
        raw_text = (char*)malloc(str_len);
        if (!raw_text)
        {
            throw std::bad_alloc();
        }
        memcpy(raw_text, str, len);
        tokens_batch.reserve(MAX_WORDS_SIZE);
    }

    std::vector<Token> get_batch()
    {
        tokens_batch.clear();

        LexerContext lctx;
        slovorez_lexer_init(&lctx);
        while (pos <= len && tokens_batch.size() != MAX_WORDS_SIZE)
        {
            if (slovorez_lexer_token_get(&lctx, (unsigned char)raw_text[pos++]))
            {
                tokens_batch.push_back(lctx.tokens.back());
            }
        }
        return tokens_batch;
    }

    ~FromTextSentencer()
    {
        if (raw_text)
        {
            free(raw_text);
            raw_text = nullptr;
        }
    }
};

class FromFileSentencer {
private:
    FILE* f = nullptr;
    std::vector<Token> tokens_batch;

public:
    FromFileSentencer(const std::string& fpath)
    {
        f = fopen(fpath.c_str(), "r");
    }

    bool is_fopen()
    {
        return f != nullptr;
    }

    std::vector<Token> get_batch()
    {
        if (f == nullptr)
        {
            return {};
        }
        tokens_batch.clear();

        LexerContext lctx;
        slovorez_lexer_init(&lctx);
        int c;
        while ((c = fgetc(f)) != EOF && tokens_batch.size() != MAX_WORDS_SIZE)
        {
            if (slovorez_lexer_token_get(&lctx, (unsigned char)c))
            {
                tokens_batch.push_back(lctx.tokens.back());
            }
        }
        return tokens_batch;
    }

    ~FromFileSentencer()
    {
        if (f != nullptr)
        {
            fclose(f);
        }
    }
};

PYBIND11_MODULE(slovorezCXX, m)
{
    py::class_<UTF8Char>(m, "UTF8Char")
        .def(py::init<>())
        .def_property_readonly("data", [](const UTF8Char& self)
            {
                return py::bytes(reinterpret_cast<const char*>(self.data), self.size);
            }
        )
        .def_property_readonly("char", [](const UTF8Char& self)
            {
                return std::string(reinterpret_cast<const char*>(self.data), self.size);
            }
        )
        .def_readwrite("size", &UTF8Char::size)
        ;

    py::enum_<TokenType>(m, "TokenType")
        .value("NOTTKN", TokenType::NOTTKN)
        .value("WRDSPC", TokenType::WRDSPC)
        .value("NWLINE", TokenType::NWLINE)
        .value("ENWORD", TokenType::ENWORD)
        .value("NUMBER", TokenType::NUMBER)
        .value("RUWORD", TokenType::RUWORD)
        .value("PNCTTN", TokenType::PNCTTN)
        .value("UNKNWN", TokenType::UNKNWN)
        .export_values()
        ;

    py::class_<Token>(m, "Token")
        .def(py::init<>())
        .def_readwrite("type", &Token::type)
        .def_readwrite("size", &Token::size)
        .def_property_readonly("data", [](const Token &self)
            {
                py::list list;
                for (int i = 0; i < self.size; ++i)
                {
                    list.append(py::cast(self.data[i]));
                }
                return list;
            }
        )
        .def("get_str", [](const Token &self)
            {
                std::string str;
                str.reserve(self.size * 4);
                for (int i = 0; i < self.size; ++i)
                {
                    str.append(reinterpret_cast<const char*>(self.data[i].data), self.data[i].size);
                }
                return str;
            }
        )
        ;

    py::class_<FromTextSentencer>(m, "FTSentencer")
        .def(py::init([](const std::string& s)
                {
                    return new FromTextSentencer(s.data(), s.size());
                }
            ),
            py::arg("text")
        )
        .def("get_batch", &FromTextSentencer::get_batch)
        ;

    py::class_<FromFileSentencer>(m, "FFSentencer")
        .def(py::init<const std::string&>(), py::arg("fpath"))
        .def("is_fopen", &FromFileSentencer::is_fopen)
        .def("get_batch", &FromFileSentencer::get_batch)
        ;
}
