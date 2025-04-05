if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

file(GLOB_RECURSE SRC_FILES
    "*.cu"
    "*.cpp"
)

foreach(CU_FILE IN LISTS SRC_FILES)
    get_filename_component(FILE_NAME ${CU_FILE} NAME_WE)
    add_executable(${FILE_NAME} ${CU_FILE})
    target_link_libraries(${FILE_NAME} PRIVATE
        CUDA::cudart
        CUDA::cublas
    )
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS " build ${CU_FILE} in DEBUG MODE")
        target_compile_options(${FILE_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-G -O0>
            $<$<COMPILE_LANGUAGE:CXX>:-g -O0>
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        message(STATUS " build ${CU_FILE} in RELEASE MODE")
        target_compile_options(${FILE_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
        )
    endif()
endforeach()