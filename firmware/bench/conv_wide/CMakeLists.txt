# Wide-conv fairness benchmark (FINDINGS F17): CMSIS-NN vs mafc's weights-as-code on a 32 -> 32 channel 3x3 conv.
# Generate the data and kernels first:  train/.venv/Scripts/python.exe harness/bench_conv.py gen
# Uses the CMSIS-NN checkout of the baseline (firmware/baselines/cmsis_nn/third_party/CMSIS-NN).

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

project(bench_conv C CXX ASM)

pico_sdk_init()

set(CMSISNN ${CMAKE_CURRENT_LIST_DIR}/../../baselines/cmsis_nn/third_party/CMSIS-NN)
if (NOT EXISTS ${CMSISNN}/Include/arm_nnfunctions.h)
    message(FATAL_ERROR "CMSIS-NN not found in ${CMSISNN} (see notes/cmsis-nn-baseline.md)")
endif()
file(GLOB CMSISNN_SRC
    ${CMSISNN}/Source/ConvolutionFunctions/*_s8*.c
    ${CMSISNN}/Source/NNSupportFunctions/*.c)
add_library(cmsisnn STATIC ${CMSISNN_SRC})
target_include_directories(cmsisnn PUBLIC ${CMSISNN}/Include)
target_compile_definitions(cmsisnn PUBLIC CMSIS_NN_USE_SINGLE_ROUNDING)
target_compile_options(cmsisnn PRIVATE -O3 -w)

option(RUN_FROM_RAM "Copy the whole program (code and data) to SRAM at boot" OFF)
set(BUILD_NAME bench_conv)
if (RUN_FROM_RAM)
    string(APPEND BUILD_NAME "_ram")
endif()

add_executable(bench_conv main.c gen/bench_code.c)
target_include_directories(bench_conv PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(bench_conv PRIVATE BUILD_NAME="${BUILD_NAME}")
target_compile_options(bench_conv PRIVATE -O3)
pico_set_program_name(bench_conv "bench_conv")
pico_enable_stdio_uart(bench_conv 0)
pico_enable_stdio_usb(bench_conv 1)
target_link_libraries(bench_conv pico_stdlib hardware_clocks cmsisnn)
if (RUN_FROM_RAM)
    pico_set_binary_type(bench_conv copy_to_ram)
endif()
pico_add_extra_outputs(bench_conv)
