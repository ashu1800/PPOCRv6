cmake_minimum_required(VERSION 3.24)

project(PPOCRv6 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PPOCR_WITH_NCNN "Build with ncnn OCR backend" ON)
option(PPOCR_BUILD_TESTS "Build unit tests" ON)

add_library(ppocr_core
    src/base64.cpp
    src/config.cpp
    src/http_server.cpp
    src/logger.cpp
    src/ocr_engine.cpp
    src/ocr_postprocess.cpp
    src/request_queue.cpp
)

target_include_directories(ppocr_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(ppocr_core
    PUBLIC
        NOMINMAX
        WIN32_LEAN_AND_MEAN
)

if(MSVC)
    target_compile_options(ppocr_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(ppocr_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_package(nlohmann_json CONFIG QUIET)
if(nlohmann_json_FOUND)
    target_link_libraries(ppocr_core PUBLIC nlohmann_json::nlohmann_json)
else()
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
    if(NOT NLOHMANN_JSON_INCLUDE_DIR)
        message(FATAL_ERROR "nlohmann/json.hpp not found. Install nlohmann_json or set CMAKE_PREFIX_PATH.")
    endif()
    target_include_directories(ppocr_core PUBLIC ${NLOHMANN_JSON_INCLUDE_DIR})
endif()

find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc)
if(OpenCV_FOUND)
    target_link_libraries(ppocr_core PUBLIC ${OpenCV_LIBS})
    target_compile_definitions(ppocr_core PUBLIC PPOCR_WITH_OPENCV=1)
else()
    find_path(OPENCV_INCLUDE_DIR opencv2/opencv.hpp)
    find_library(OPENCV_CORE_LIB NAMES opencv_core opencv_core4100)
    find_library(OPENCV_IMGCODECS_LIB NAMES opencv_imgcodecs opencv_imgcodecs4100)
    find_library(OPENCV_IMGPROC_LIB NAMES opencv_imgproc opencv_imgproc4100)
    if(OPENCV_INCLUDE_DIR AND OPENCV_CORE_LIB AND OPENCV_IMGCODECS_LIB AND OPENCV_IMGPROC_LIB AND MSVC)
        target_include_directories(ppocr_core PUBLIC ${OPENCV_INCLUDE_DIR})
        target_link_libraries(ppocr_core PUBLIC ${OPENCV_CORE_LIB} ${OPENCV_IMGCODECS_LIB} ${OPENCV_IMGPROC_LIB})
        target_compile_definitions(ppocr_core PUBLIC PPOCR_WITH_OPENCV=1)
    else()
        target_compile_definitions(ppocr_core PUBLIC PPOCR_WITH_OPENCV=0)
        message(WARNING "OpenCV not found or not ABI-compatible with the selected compiler. OCR image decode will return an error until OpenCV is available.")
    endif()
endif()

if(PPOCR_WITH_NCNN)
    find_package(ncnn CONFIG QUIET)
    if(ncnn_FOUND)
        target_link_libraries(ppocr_core PUBLIC ncnn)
        target_compile_definitions(ppocr_core PUBLIC PPOCR_WITH_NCNN=1)
    else()
        target_compile_definitions(ppocr_core PUBLIC PPOCR_WITH_NCNN=0)
        message(WARNING "ncnn package not found. Building CPU placeholder backend; install ncnn to enable real PPOCR inference.")
    endif()
else()
    target_compile_definitions(ppocr_core PUBLIC PPOCR_WITH_NCNN=0)
endif()

if(WIN32)
    target_link_libraries(ppocr_core PUBLIC ws2_32)
endif()

add_executable(PPOCRv6 src/main.cpp)
target_link_libraries(PPOCRv6 PRIVATE ppocr_core)

if(PPOCR_BUILD_TESTS)
    enable_testing()
    add_executable(ppocr_tests
        tests/test_main.cpp
    )
    target_link_libraries(ppocr_tests PRIVATE ppocr_core)
    add_test(NAME ppocr_tests COMMAND ppocr_tests)
endif()
