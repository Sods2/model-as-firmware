# Idea C benchmark (FINDINGS F21): DMA streaming of weights from flash vs reading them through the XIP cache.
# Generate the data first:  python harness/bench_stream.py gen
# Always a flash build (the weights must stay in flash); the kernels run from SRAM (__not_in_flash_func).

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.3.1)
set(toolchainVersion 15_2_Rel1)
set(picotoolVersion 2.3.1)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico2 CACHE STRING "Board type")

include(${CMAKE_CURRENT_LIST_DIR}/../../rp2350/hello/pico_sdk_import.cmake)

project(bench_stream C CXX ASM)

pico_sdk_init()

add_executable(bench_stream main.c)
target_include_directories(bench_stream PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(bench_stream PRIVATE BUILD_NAME="bench_stream")
target_compile_options(bench_stream PRIVATE -O3)
pico_set_program_name(bench_stream "bench_stream")
pico_enable_stdio_uart(bench_stream 0)
pico_enable_stdio_usb(bench_stream 1)
target_link_libraries(bench_stream pico_stdlib hardware_clocks hardware_dma)
pico_add_extra_outputs(bench_stream)
