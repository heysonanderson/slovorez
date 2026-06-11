pybind11_add_module(slovorezCXX pybinder.cc tokenizer.cc lexer.cc unicode_trie.cc token.h utf8_decoder.h)

if(MINGW)
    target_link_options(slovorezCXX PRIVATE "-static-libgcc" "-static-libstdc++" "-static")
endif()

set_target_properties(slovorezCXX PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/src"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/src"
)