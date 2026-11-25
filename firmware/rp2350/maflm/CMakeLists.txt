# MafLM (Phase 3 tiny story model) on the Pico 2: the integer LM runtime (lm_rt.c) with compiler-generated model
# data (gen/maflm_<MODEL>.c, from compiler/compile_lm.py) and the serial harness (pico_main.c).
#
#   -DMODEL=lm_tree           gen/maflm_<MODEL>.{c,h}
#   -DRUN_FROM_RAM=ON         copy everything (code + weights) to SRAM at boot; otherwise weights stay in flash
#   -DLM_PROFILE=ON           per-stage cycle counters ('Q' command)
#   -DDUAL_CORE=ON            split matvec rows, table builds and attention heads across both cores

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
set(MODEL lm_tree CACHE STRING "Model (gen/maflm_<MODEL>.c)")

include(${CMAKE_CURRENT_LIST_DIR}/../hello/pico_sdk_import.cmake)

project(maflm C CXX ASM)

pico_sdk_init()

option(RUN_FROM_RAM "Copy the whole program (code and weights) to SRAM at boot" OFF)
option(LM_PROFILE "Per-stage cycle counters" OFF)
option(DUAL_CORE "Split rows, tables and attention heads across both M33 cores" OFF)
set(BUILD_NAME "maflm_v1_${MODEL}")
if (RUN_FROM_RAM)
    string(APPEND BUILD_NAME "_ram")
else()
    string(APPEND BUILD_NAME "_flash")
endif()
if (DUAL_CORE)
    string(APPEND BUILD_NAME "_dual")
endif()
if (LM_PROFILE)
    string(APPEND BUILD_NAME "_prof")
endif()

add_executable(maflm pico_main.c lm_rt.c gen/maflm_${MODEL}.c)
target_include_directories(maflm PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/gen)
target_compile_definitions(maflm PRIVATE
    LM_GEN_HEADER="maflm_${MODEL}.h"
    BUILD_NAME="${BUILD_NAME}"
    PICO_STACK_SIZE=0x1000)
if (LM_PROFILE)
    target_compile_definitions(maflm PRIVATE LM_PROFILE=1)
endif()
if (DUAL_CORE)
    target_compile_definitions(maflm PRIVATE LM_DUAL=1)
    target_link_libraries(maflm pico_multicore)
endif()

pico_set_program_name(maflm "maflm")
pico_enable_stdio_uart(maflm 0)
pico_enable_stdio_usb(maflm 1)
target_link_libraries(maflm pico_stdlib hardware_clocks)

if (RUN_FROM_RAM)
    pico_set_binary_type(maflm copy_to_ram)
endif()

pico_add_extra_outputs(maflm)
