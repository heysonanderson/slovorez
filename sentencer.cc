#include <cstdio>
#include <cstring>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "text_lexer.h"

namespace py = pybind11;

constexpr size_t DEFAULT_BATCH_SIZE = 1048576;

class FromTextSentencer {
private:
    LexerContext lctx;
    char* raw_text = nullptr;
    char* str_buf = nullptr;
    TokenType* ttype_buf = nullptr;
    size_t len = 0;
    size_t pos = 0;
    size_t batch_size = DEFAULT_BATCH_SIZE;

public:
    FromTextSentencer(const char* str, size_t str_len) : len(str_len), pos(0)
    {
        raw_text = (char*)malloc(str_len);
        str_buf = (char*)malloc((512 * batch_size + batch_size) * sizeof(char));
        ttype_buf = (TokenType*)malloc(batch_size * sizeof(TokenType));
        memcpy(raw_text, str, len);
        slovorez_lexer_init(&lctx);
    }

    void set_batch_size(size_t new_size)
    {
        batch_size = new_size;
        str_buf = (char*)realloc(str_buf, (512 * batch_size + batch_size) * sizeof(char));
        ttype_buf = (TokenType*)realloc(ttype_buf, batch_size * sizeof(TokenType));
    }

    py::dict get_batch()
    {
        size_t str_size = 0;
        size_t token_idx = 0;
        while (pos <= len && token_idx <= batch_size)
        {
            if (slovorez_lexer_token_get(&lctx, (unsigned char)raw_text[pos++]))
            {
                const Token& token = lctx.rtoken;
                for (int i = 0; i < token.size; ++i)
                {
                    memcpy(str_buf + str_size, token.data[i].data, token.data[i].size);
                    str_size += token.data[i].size;
                }
                str_buf[str_size++] = '\0';
                ttype_buf[token_idx] = token.type;
                token_idx += 1;
            }
        }
        if (token_idx == 0)
        {
            return py::dict();
        }
        py::dict outbuf;
        outbuf["text"] = py::str(str_buf, str_size);
        outbuf["types"] = py::array_t<int>(
            { (size_t)token_idx },
            { sizeof(int) },
            reinterpret_cast<int*>(ttype_buf),
            py::cast(this)
        );
        return outbuf;
    }

    ~FromTextSentencer()
    {
        if (raw_text != nullptr)
        {
            free(raw_text);
            raw_text = nullptr;
        }
        if (str_buf != nullptr)
        {
            free(str_buf);
            str_buf = nullptr;
        }
        if (ttype_buf != nullptr)
        {
            free(ttype_buf);
            ttype_buf = nullptr;
        }
    }
};

class FromFileSentencer {
private:
    LexerContext lctx;
    FILE* f = nullptr;
    char* str_buf = nullptr;
    TokenType* ttype_buf = nullptr;
    size_t batch_size = DEFAULT_BATCH_SIZE;

public:
    FromFileSentencer(const std::string& fpath)
    {
        f = fopen(fpath.c_str(), "r");
        str_buf = (char*)malloc((512 * batch_size + batch_size) * sizeof(char));
        ttype_buf = (TokenType*)malloc(batch_size * sizeof(TokenType));
        slovorez_lexer_init(&lctx);
    }

    void set_batch_size(size_t new_size)
    {
        batch_size = new_size;
        str_buf = (char*)realloc(str_buf, (512 * batch_size + batch_size) * sizeof(char));
        ttype_buf = (TokenType*)realloc(ttype_buf, batch_size * sizeof(TokenType));
    }

    bool is_fopen()
    {
        return f != nullptr;
    }

    py::dict get_batch()
    {
        if (f == nullptr)
        {
            return py::dict();
        }
        size_t str_size = 0;
        size_t token_idx = 0;
        int c;
        while ((c = fgetc(f)) != EOF && token_idx <= batch_size)
        {
            if (slovorez_lexer_token_get(&lctx, (unsigned char)c))
            {
                const Token& token = lctx.rtoken;
                for (int i = 0; i < token.size; ++i)
                {
                    memcpy(str_buf + str_size, token.data[i].data, token.data[i].size);
                    str_size += token.data[i].size;
                }
                str_buf[str_size++] = '\0';
                ttype_buf[token_idx] = token.type;
                token_idx += 1;
            }
        }
        if (token_idx == 0)
        {
            return py::dict();
        }
        py::dict outbuf;
        outbuf["text"] = py::str(str_buf, str_size);
        outbuf["types"] = py::array_t<int>(
            { (size_t)token_idx },
            { sizeof(int) },
            reinterpret_cast<int*>(ttype_buf),
            py::cast(this)
        );
        return outbuf;
    }

    ~FromFileSentencer()
    {
        if (f != nullptr)
        {
            fclose(f);
        }
        if (str_buf != nullptr)
        {
            free(str_buf);
            str_buf = nullptr;
        }
        if (ttype_buf != nullptr)
        {
            free(ttype_buf);
            ttype_buf = nullptr;
        }
    }
};

typedef struct FromTextStream {
    FromTextSentencer &sentencer;
    FromTextStream(FromTextSentencer& s) : sentencer(s) {}
} FromTextStream;

typedef struct FromFileStream {
    FromFileSentencer &sentencer;
    FromFileStream(FromFileSentencer& s) : sentencer(s) {}
} FromFileStream;

PYBIND11_MODULE(slovorezCXX, m)
{
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

    py::class_<FromTextStream>(m, "fts_stream")
        .def("__iter__", [](FromTextStream &self) { return self; })
        .def("__next__", [](FromTextStream &self)
            {
                py::dict batch = self.sentencer.get_batch();
                if (batch.empty())
                {
                    throw py::stop_iteration();
                }
                return batch;
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
        .def("set_batch_size", &FromTextSentencer::set_batch_size)
        .def("get_batch", &FromTextSentencer::get_batch)
        .def_property_readonly("stream", [](FromTextSentencer& self)
            {
                return FromTextStream(self);
            }
        )
    ;

    py::class_<FromFileStream>(m, "ffs_stream")
        .def("__iter__", [](FromFileStream &self) { return self; })
        .def("__next__", [](FromFileStream &self)
            {
                py::dict batch = self.sentencer.get_batch();
                if (batch.empty())
                {
                    throw py::stop_iteration();
                }
                return batch;
            }
        )
    ;

    py::class_<FromFileSentencer>(m, "FFSentencer")
        .def(py::init<const std::string&>(), py::arg("fpath"))
        .def("is_fopen", &FromFileSentencer::is_fopen)
        .def("set_batch_size", &FromFileSentencer::set_batch_size)
        .def("get_batch", &FromFileSentencer::get_batch)
        .def_property_readonly("stream", [](FromFileSentencer& self)
            {
                return FromFileStream(self);
            }
        )
    ;
}
