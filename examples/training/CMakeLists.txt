set(TARGET llama-finetune)
add_executable(${TARGET} finetune.cpp parquet_dataset.cpp)
install(TARGETS ${TARGET} RUNTIME)


if(LLAMA_PARQUET)
    find_package(Arrow REQUIRED)
    find_package(Parquet REQUIRED)
    add_definitions(-DLLAMA_PARQUET)
endif()

if(LLAMA_PARQUET)
    target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} Arrow::arrow_shared Parquet::parquet_shared)
else()
    target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_11)


set(TARGET convert-to-train-gguf)
add_executable(${TARGET} convert-to-train-gguf.cpp)
install(TARGETS ${TARGET} RUNTIME)


if(LLAMA_PARQUET)
    find_package(Arrow REQUIRED)
    find_package(Parquet REQUIRED)
    add_definitions(-DLLAMA_PARQUET)
endif()

if(LLAMA_PARQUET)
    target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} Arrow::arrow_shared Parquet::parquet_shared)
else()
    target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_11)
