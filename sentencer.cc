#include <cstdio>
#include <cstring>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "text_lexer.h"

namespace py = pybind11;
using namespace py::literals;

constexpr size_t DEFAULT_BATCH_SIZE = 1048576;

class FromTextSentencer {
private:
    LexerContext lctx;
    char* raw_text = nullptr;
    size_t text_len = 0;
    size_t text_pos = 0;
    char* batch_str_buf = nullptr;
    TokenType* batch_types_buf = nullptr;
    size_t batch_size = DEFAULT_BATCH_SIZE;
    uint64_t filter_mask = 0xFFFFFFFFFFFFFFFF;

public:
    FromTextSentencer(const char* str, size_t str_len) : text_len(str_len), text_pos(0)
    {
        this->raw_text = (char*)malloc(str_len);
        this->batch_str_buf = (char*)malloc((512 * this->batch_size + this->batch_size) * sizeof(char));
        this->batch_types_buf = (TokenType*)malloc(this->batch_size * sizeof(TokenType));
        memcpy(this->raw_text, str, str_len);
        slovorez_lexer_init(&this->lctx);
    }

    void set_batch_size(size_t batch_size)
    {
        this->batch_size = batch_size;
        this->batch_str_buf = (char*)realloc(this->batch_str_buf, (512 * this->batch_size + this->batch_size) * sizeof(char));
        this->batch_types_buf = (TokenType*)realloc(this->batch_types_buf, this->batch_size * sizeof(TokenType));
    }

    void set_filter(uint64_t filter_mask)
    {
        this->filter_mask = filter_mask;
    }

    py::dict get_batch()
    {
        size_t batch_str_size = 0;
        size_t batch_token_idx = 0;
        while (this->text_pos <= this->text_len && batch_token_idx <= this->batch_size)
        {
            if (slovorez_lexer_token_get(&this->lctx, (unsigned char)this->raw_text[this->text_pos++]))
            {
                const Token& token = this->lctx.rtoken;
                if (static_cast<uint64_t>(token.type) & this->filter_mask)
                {
                    for (int i = 0; i < token.size; ++i)
                    {
                        memcpy(this->batch_str_buf + batch_str_size, token.data[i].data, token.data[i].size);
                        batch_str_size += token.data[i].size;
                    }
                    this->batch_str_buf[batch_str_size++] = '\0';
                    this->batch_types_buf[batch_token_idx++] = token.type;
                }
            }
        }
        if (batch_token_idx == 0)
        {
            return py::dict();
        }
        py::dict outbuf;
        outbuf["text"_s] = py::str(this->batch_str_buf, batch_str_size);
        outbuf["types"_s] = py::array_t<uint64_t>(
            { (size_t)batch_token_idx },
            { sizeof(uint64_t) },
            reinterpret_cast<uint64_t*>(this->batch_types_buf),
            py::cast(this)
        );
        return outbuf;
    }

    ~FromTextSentencer()
    {
        if (this->raw_text != nullptr)
        {
            free(this->raw_text);
            this->raw_text = nullptr;
        }
        if (this->batch_str_buf != nullptr)
        {
            free(this->batch_str_buf);
            this->batch_str_buf = nullptr;
        }
        if (this->batch_types_buf != nullptr)
        {
            free(this->batch_types_buf);
            this->batch_types_buf = nullptr;
        }
    }
};

class FromFileSentencer {
private:
    LexerContext lctx;
    FILE* f = nullptr;
    char* batch_str_buf = nullptr;
    TokenType* batch_types_buf = nullptr;
    size_t batch_size = DEFAULT_BATCH_SIZE;
    uint64_t filter_mask = 0xFFFFFFFFFFFFFFFF;

public:
    FromFileSentencer(const std::string& fpath)
    {
        this->f = fopen(fpath.c_str(), "r");
        this->batch_str_buf = (char*)malloc((512 * this->batch_size + this->batch_size) * sizeof(char));
        this->batch_types_buf = (TokenType*)malloc(this->batch_size * sizeof(TokenType));
        slovorez_lexer_init(&this->lctx);
    }

    void set_batch_size(size_t batch_size)
    {
        this->batch_size = batch_size;
        this->batch_str_buf = (char*)realloc(this->batch_str_buf, (512 * this->batch_size + this->batch_size) * sizeof(char));
        this->batch_types_buf = (TokenType*)realloc(this->batch_types_buf, this->batch_size * sizeof(TokenType));
    }

    void set_filter(uint64_t filter_mask)
    {
        this->filter_mask = filter_mask;
    }

    bool is_fopen()
    {
        return this->f != nullptr;
    }

    py::dict get_batch()
    {
        if (this->f == nullptr)
        {
            return py::dict();
        }
        size_t batch_str_size = 0;
        size_t batch_token_idx = 0;
        int c;
        while ((c = fgetc(this->f)) != EOF && batch_token_idx <= this->batch_size)
        {
            if (slovorez_lexer_token_get(&this->lctx, (unsigned char)c))
            {
                const Token& token = this->lctx.rtoken;
                if (static_cast<uint64_t>(token.type) & this->filter_mask)
                {
                    for (int i = 0; i < token.size; ++i)
                    {
                        memcpy(this->batch_str_buf + batch_str_size, token.data[i].data, token.data[i].size);
                        batch_str_size += token.data[i].size;
                    }
                    this->batch_str_buf[batch_str_size++] = '\0';
                    this->batch_types_buf[batch_token_idx++] = token.type;
                }
            }
        }
        if (batch_token_idx == 0)
        {
            return py::dict();
        }
        py::dict outbuf;
        outbuf["text"_s] = py::str(this->batch_str_buf, batch_str_size);
        outbuf["types"_s] = py::array_t<uint64_t>(
            { (size_t)batch_token_idx },
            { sizeof(uint64_t) },
            reinterpret_cast<uint64_t*>(this->batch_types_buf),
            py::cast(this)
        );
        return outbuf;
    }

    ~FromFileSentencer()
    {
        if (this->f != nullptr)
        {
            fclose(this->f);
        }
        if (this->batch_str_buf != nullptr)
        {
            free(this->batch_str_buf);
            this->batch_str_buf = nullptr;
        }
        if (this->batch_types_buf != nullptr)
        {
            free(this->batch_types_buf);
            this->batch_types_buf = nullptr;
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
    py::enum_<TokenType>(m, "TokenType", py::arithmetic())
        .value("NOTTKN", TokenType::NOTTKN)
        .value("WRDSPC", TokenType::WRDSPC)
        .value("NWLINE", TokenType::NWLINE)
        .value("ENWORD", TokenType::ENWORD)
        .value("NUMBER", TokenType::NUMBER)
        .value("RUWORD", TokenType::RUWORD)
        .value("PNCTTN", TokenType::PNCTTN)
        .value("UNKNWN", TokenType::UNKNWN)
        .export_values()
        .def("__or__", [](TokenType a, TokenType b)
            {
                return static_cast<uint64_t>(a) | static_cast<uint64_t>(b);
            }
        )
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
        .def("set_filter", &FromTextSentencer::set_filter)
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
        .def("set_filter", &FromFileSentencer::set_filter)
        .def("get_batch", &FromFileSentencer::get_batch)
        .def_property_readonly("stream", [](FromFileSentencer& self)
            {
                return FromFileStream(self);
            }
        )
    ;
}
