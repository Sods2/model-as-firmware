# CMSIS-NN int8 baseline on the Pico 2 (Phase 0 step 5).
# Needs CMSIS-NN in third_party/CMSIS-NN (untracked; see notes/cmsis-nn-baseline.md for the pinned commit).
# Model chosen with -DMODEL=tern80 (default), tern80z70 or tern80z90; export it first with harness/export_cmsis.py.

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
set(MODEL tern80 CACHE STRING "Model (header model_s8_<MODEL>.h in this folder)")

include(${CMAKE_CURRENT_LIST_DIR}/../../rp2350/hello/pico_sdk_import.cmake)

project(cmsis_nn C CXX ASM)

pico_sdk_init()

set(CMSISNN ${CMAKE_CURRENT_LIST_DIR}/third_party/CMSIS-NN)
if (NOT EXISTS ${CMSISNN}/Include/arm_nnfunctions.h)
    message(FATAL_ERROR "CMSIS-NN not found in ${CMSISNN}; clone it there (see notes/cmsis-nn-baseline.md)")
endif()
file(GLOB CMSISNN_SRC
    ${CMSISNN}/Source/ConvolutionFunctions/*_s8*.c
    ${CMSISNN}/Source/FullyConnectedFunctions/*_s8*.c
    ${CMSISNN}/Source/PoolingFunctions/*_s8*.c
    ${CMSISNN}/Source/NNSupportFunctions/*.c)

add_library(cmsisnn STATIC ${CMSISNN_SRC})
target_include_directories(cmsisnn PUBLIC ${CMSISNN}/Include)
target_compile_definitions(cmsisnn PUBLIC CMSIS_NN_USE_SINGLE_ROUNDING)
target_compile_options(cmsisnn PRIVATE -O3 -w)

option(RUN_FROM_RAM "Copy the whole program (code and weights) to SRAM at boot" OFF)
option(PROFILE_STAGES "Compile in per-stage cycle counters (costs a few %)" OFF)
# fc1 with CMSIS-NN's s4 kernel (ternary weights packed 4 bits each; header from harness/export_cmsis_s4.py).
# fc2/fc3 stay s8: their BitNetMCU 4-bit levels (+-1..+-15, odd) don't fit a two's-complement int4.
option(FC1_S4 "fc1 as arm_fully_connected_s4 (model_s4fc1_<MODEL>.h)" OFF)
if (FC1_S4)
    file(GLOB CMSISNN_S4_SRC ${CMSISNN}/Source/FullyConnectedFunctions/arm_fully_connected_s4.c)
    target_sources(cmsisnn PRIVATE ${CMSISNN_S4_SRC})
    set(MODEL_HEADER model_s4fc1_${MODEL}.h)
else()
    set(MODEL_HEADER model_s8_${MODEL}.h)
endif()
set(BUILD_NAME cmsis_nn)
if (FC1_S4)
    string(APPEND BUILD_NAME "_s4fc1")
endif()
if (RUN_FROM_RAM)
    string(APPEND BUILD_NAME "_ram")
endif()
if (PROFILE_STAGES)
    string(APPEND BUILD_NAME "_prof")
endif()

add_executable(cmsis_nn ../ref_c/pico_main.c cmsis_infer.c)
target_include_directories(cmsis_nn PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../ref_c)
target_compile_definitions(cmsis_nn PRIVATE
    S8_MODEL_HEADER="${MODEL_HEADER}"
    BUILD_NAME="${BUILD_NAME}")
if (FC1_S4)
    target_compile_definitions(cmsis_nn PRIVATE FC1_S4=1)
endif()

pico_set_program_name(cmsis_nn "cmsis_nn")
pico_set_program_version(cmsis_nn "0.1")

pico_enable_stdio_uart(cmsis_nn 0)
pico_enable_stdio_usb(cmsis_nn 1)

target_link_libraries(cmsis_nn pico_stdlib hardware_clocks cmsisnn)

if (RUN_FROM_RAM)
    pico_set_binary_type(cmsis_nn copy_to_ram)
endif()
if (PROFILE_STAGES)
    target_compile_definitions(cmsis_nn PRIVATE REF_PROFILE=1)
endif()

pico_add_extra_outputs(cmsis_nn)
