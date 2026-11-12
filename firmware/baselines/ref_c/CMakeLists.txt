# Plain-C reference inference on the Pico 2 (Phase 0 step 4).
# Model header from train/export/, chosen with -DMODEL=tern80 (default), tern80z70 or tern80z90.

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
set(MODEL tern80 CACHE STRING "Model header in train/export/ (without .h)")

include(${CMAKE_CURRENT_LIST_DIR}/../../rp2350/hello/pico_sdk_import.cmake)

project(ref_c C CXX ASM)

pico_sdk_init()

option(RUN_FROM_RAM "Copy the whole program (code and weights) to SRAM at boot" OFF)
option(PROFILE_STAGES "Compile in per-stage cycle counters (costs a few %)" OFF)
set(BUILD_NAME ref_c)
if (RUN_FROM_RAM)
    string(APPEND BUILD_NAME "_ram")
endif()
if (PROFILE_STAGES)
    string(APPEND BUILD_NAME "_prof")
endif()

add_executable(ref_c pico_main.c ref_infer.c)

target_include_directories(ref_c PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../../train/export)
target_compile_definitions(ref_c PRIVATE
    MODEL_HEADER="${MODEL}.h"
    BUILD_NAME="${BUILD_NAME}")

pico_set_program_name(ref_c "ref_c")
pico_set_program_version(ref_c "0.1")

pico_enable_stdio_uart(ref_c 0)
pico_enable_stdio_usb(ref_c 1)

target_link_libraries(ref_c pico_stdlib hardware_clocks)

if (RUN_FROM_RAM)
    pico_set_binary_type(ref_c copy_to_ram)
endif()
if (PROFILE_STAGES)
    target_compile_definitions(ref_c PRIVATE REF_PROFILE=1)
endif()

pico_add_extra_outputs(ref_c)
