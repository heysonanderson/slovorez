#include <cstdio>
#include <cstring>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "token.h"
#include "tokenizer.h"

namespace py = pybind11;
using namespace py::literals;

constexpr size_t DEFAULT_BATCH_SIZE = 65536;
constexpr size_t DEFAULT_TOKEN_MIN_LEN = 0;
constexpr size_t DEFAULT_TOKEN_MAX_LEN = 512;
constexpr uint64_t NO_FILTER_MASK = 0xFFFFFFFFFFFFFFFF;

typedef struct {
    char* str = nullptr;
    TokenType* types = nullptr;
    size_t str_size;
    size_t token_idx;
} TokenizerBatchBuffer;

typedef struct {
    size_t batch_size = DEFAULT_BATCH_SIZE;
    size_t token_min_len = DEFAULT_TOKEN_MIN_LEN;
    size_t token_max_len = DEFAULT_TOKEN_MAX_LEN;
    uint64_t filter_mask = NO_FILTER_MASK;
} TokenizerConfig;

class FromTextTokenizer {
private:
    TokenizerContext tctx;
    char* raw_text = nullptr;
    size_t text_len = 0;
    size_t text_pos = 0;
    TokenizerBatchBuffer batch_buffer;
    TokenizerConfig config;

    inline void _push_token_to_batch(const Token& token)
    {
        const bool allowed_type = slovorez_token_filter_match(token.type, this->config.filter_mask);
        const bool allowed_size = this->config.token_min_len <= token.size && token.size <= this->config.token_max_len;
        if (allowed_type && allowed_size)
        {
            for (int i = 0; i < token.size; ++i)
            {
                memcpy(this->batch_buffer.str + this->batch_buffer.str_size, token.data[i].bytes, token.data[i].size);
                this->batch_buffer.str_size += token.data[i].size;
            }
            this->batch_buffer.str[this->batch_buffer.str_size++] = '\0';
            this->batch_buffer.types[this->batch_buffer.token_idx++] = token.type;
        }
    }

public:
    FromTextTokenizer(const char* str, size_t str_len) : text_len(str_len), text_pos(0)
    {
        this->raw_text = (char*)malloc(str_len);
        memcpy(this->raw_text, str, str_len);
        this->batch_buffer.str = (char*)malloc((512 * this->config.batch_size + this->config.batch_size) * sizeof(char));
        this->batch_buffer.types = (TokenType*)malloc(this->config.batch_size * sizeof(TokenType));
        slovorez_tokenizer_init(&this->tctx);
    }

    void set_batch_size(size_t batch_size)
    {
        this->config.batch_size = batch_size;
        this->batch_buffer.str = (char*)realloc(this->batch_buffer.str, (512 * this->config.batch_size + this->config.batch_size) * sizeof(char));
        this->batch_buffer.types = (TokenType*)realloc(this->batch_buffer.types, this->config.batch_size * sizeof(TokenType));
    }

    void set_filter(uint64_t filter_mask)
    {
        this->config.filter_mask = filter_mask;
    }

    void set_token_min_len(size_t token_min_len)
    {
        this->config.token_min_len = token_min_len;
    }

    void set_token_max_len(size_t token_max_len)
    {
        this->config.token_max_len = token_max_len;
    }

    py::dict get_batch()
    {
        this->batch_buffer.str_size = 0;
        this->batch_buffer.token_idx = 0;
        while (this->text_pos <= this->text_len && this->batch_buffer.token_idx <= this->config.batch_size)
        {
            if (slovorez_tokenizer_token_get(&this->tctx, (unsigned char)this->raw_text[this->text_pos++]))
            {
                this->_push_token_to_batch(this->tctx.rtoken);
            }
        }
        if (this->text_pos >= this->text_len && this->batch_buffer.token_idx < this->config.batch_size && slovorez_tokenizer_end(&this->tctx))
        {
            this->_push_token_to_batch(this->tctx.rtoken);
        }

        if (this->batch_buffer.token_idx == 0)
        {
            return py::dict();
        }
        py::dict outbuf;
        outbuf["text"_s] = py::str(this->batch_buffer.str, this->batch_buffer.str_size);
        outbuf["types"_s] = py::array_t<uint64_t>(
            { this->batch_buffer.token_idx },
            { sizeof(uint64_t) },
            reinterpret_cast<uint64_t*>(this->batch_buffer.types),
            py::cast(this)
        );
        return outbuf;
    }

    ~FromTextTokenizer()
    {
        if (this->raw_text != nullptr)
        {
            free(this->raw_text);
            this->raw_text = nullptr;
        }
        if (this->batch_buffer.str != nullptr)
        {
            free(this->batch_buffer.str);
            this->batch_buffer.str = nullptr;
        }
        if (this->batch_buffer.types != nullptr)
        {
            free(this->batch_buffer.types);
            this->batch_buffer.types = nullptr;
        }
    }
};

class FromFileTokenizer {
private:
    TokenizerContext tctx;
    FILE* f = nullptr;
    TokenizerBatchBuffer batch_buffer;
    TokenizerConfig config;

    inline void _push_token_to_batch(const Token& token)
    {
        const bool allowed_type = slovorez_token_filter_match(token.type, this->config.filter_mask);
        const bool allowed_size = this->config.token_min_len <= token.size && token.size <= this->config.token_max_len;
        if (allowed_type && allowed_size)
        {
            for (int i = 0; i < token.size; ++i)
            {
                memcpy(this->batch_buffer.str + this->batch_buffer.str_size, token.data[i].bytes, token.data[i].size);
                this->batch_buffer.str_size += token.data[i].size;
            }
            this->batch_buffer.str[this->batch_buffer.str_size++] = '\0';
            this->batch_buffer.types[this->batch_buffer.token_idx++] = token.type;
        }
    }

public:
    FromFileTokenizer(const std::string& fpath)
    {
        this->f = fopen(fpath.c_str(), "r");
        this->batch_buffer.str = (char*)malloc((512 * this->config.batch_size + this->config.batch_size) * sizeof(char));
        this->batch_buffer.types = (TokenType*)malloc(this->config.batch_size * sizeof(TokenType));
        slovorez_tokenizer_init(&this->tctx);
    }

    void set_batch_size(size_t batch_size)
    {
        this->config.batch_size = batch_size;
        this->batch_buffer.str = (char*)realloc(this->batch_buffer.str, (512 * this->config.batch_size + this->config.batch_size) * sizeof(char));
        this->batch_buffer.types = (TokenType*)realloc(this->batch_buffer.types, this->config.batch_size * sizeof(TokenType));
    }

    void set_filter(uint64_t filter_mask)
    {
        this->config.filter_mask = filter_mask;
    }

    void set_token_min_len(size_t token_min_len)
    {
        this->config.token_min_len = token_min_len;
    }

    void set_token_max_len(size_t token_max_len)
    {
        this->config.token_max_len = token_max_len;
    }

    bool is_fopen()
    {
        return this->f != nullptr;
    }

    py::dict get_batch()
    {
        this->batch_buffer.str_size = 0;
        this->batch_buffer.token_idx = 0;
        if (this->f == nullptr)
        {
            return py::dict();
        }
        int c;
        while ((c = fgetc(this->f)) != EOF && this->batch_buffer.token_idx < this->config.batch_size)
        {
            if (slovorez_tokenizer_token_get(&this->tctx, (unsigned char)c))
            {
                this->_push_token_to_batch(this->tctx.rtoken);
            }
        }
        if (c == EOF && this->batch_buffer.token_idx < this->config.batch_size && slovorez_tokenizer_end(&this->tctx))
        {
            this->_push_token_to_batch(this->tctx.rtoken);
        }

        if (this->batch_buffer.token_idx == 0)
        {
            return py::dict();
        }
        py::dict outbuf;
        outbuf["text"_s] = py::str(this->batch_buffer.str, this->batch_buffer.str_size);
        outbuf["types"_s] = py::array_t<uint64_t>(
            { this->batch_buffer.token_idx },
            { sizeof(uint64_t) },
            reinterpret_cast<uint64_t*>(this->batch_buffer.types),
            py::cast(this)
        );
        return outbuf;
    }

    ~FromFileTokenizer()
    {
        if (this->f != nullptr)
        {
            fclose(this->f);
        }
        if (this->batch_buffer.str != nullptr)
        {
            free(this->batch_buffer.str);
            this->batch_buffer.str = nullptr;
        }
        if (this->batch_buffer.types != nullptr)
        {
            free(this->batch_buffer.types);
            this->batch_buffer.types = nullptr;
        }
    }
};

typedef struct FromTextStream {
    FromTextTokenizer &sentencer;
    FromTextStream(FromTextTokenizer& s) : sentencer(s) {}
} FromTextStream;

typedef struct FromFileStream {
    FromFileTokenizer &sentencer;
    FromFileStream(FromFileTokenizer& s) : sentencer(s) {}
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

    py::class_<FromTextTokenizer>(m, "FTTokenizer")
        .def(py::init([](const std::string& s)
                {
                    return new FromTextTokenizer(s.data(), s.size());
                }
            ),
            py::arg("text")
        )
        .def("set_batch_size", &FromTextTokenizer::set_batch_size)
        .def("set_filter", &FromTextTokenizer::set_filter)
        .def("set_token_min_len", &FromTextTokenizer::set_token_min_len)
        .def("set_token_max_len", &FromTextTokenizer::set_token_max_len)
        .def("get_batch", &FromTextTokenizer::get_batch)
        .def_property_readonly("stream", [](FromTextTokenizer& self)
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

    py::class_<FromFileTokenizer>(m, "FFTokenizer")
        .def(py::init<const std::string&>(), py::arg("fpath"))
        .def("is_fopen", &FromFileTokenizer::is_fopen)
        .def("set_batch_size", &FromFileTokenizer::set_batch_size)
        .def("set_filter", &FromFileTokenizer::set_filter)
        .def("set_token_min_len", &FromFileTokenizer::set_token_min_len)
        .def("set_token_max_len", &FromFileTokenizer::set_token_max_len)
        .def("get_batch", &FromFileTokenizer::get_batch)
        .def_property_readonly("stream", [](FromFileTokenizer& self)
            {
                return FromFileStream(self);
            }
        )
    ;
}
