pybind11_add_module(slovorezCXX pybinder.cc tokenizer.cc lexer.cc unicode_trie.cc token.h utf8_decoder.h)

if(MINGW)
    target_link_options(slovorezCXX PRIVATE "-static-libgcc" "-static-libstdc++" "-static")
endif()

install(TARGETS slovorezCXX DESTINATION .)
install(FILES ${CMAKE_SOURCE_DIR}/src/slovorezCXX.pyi DESTINATION .)
