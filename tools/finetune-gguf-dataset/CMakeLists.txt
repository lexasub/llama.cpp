include_directories(. ../../src/)
include_directories(../dataset-converter)

add_executable(finetune-gguf finetune-gguf.cpp)
target_link_libraries(finetune-gguf PRIVATE dataset-to-gguf-lib)

