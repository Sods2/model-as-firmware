# mafc-compiled model on the Pico 2 (Phase 1). Uses the generated gen/mafc_<MODEL>.c
# (regenerate with compiler/compile.py) and the shared harness firmware ../../baselines/ref_c/pico_main.c.

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
set(MODEL tern80 CACHE STRING "Model (gen/mafc_<MODEL>.c)")

include(${CMAKE_CURRENT_LIST_DIR}/../../rp2350/hello/pico_sdk_import.cmake)

project(mafc C CXX ASM)

pico_sdk_init()

option(RUN_FROM_RAM "Copy the whole program (code and weights) to SRAM at boot" OFF)
option(PROFILE_STAGES "Compile in per-stage cycle counters (costs a few %)" OFF)
set(BUILD_NAME mafc_v0)
if (RUN_FROM_RAM)
    string(APPEND BUILD_NAME "_ram")
endif()
if (PROFILE_STAGES)
    string(APPEND BUILD_NAME "_prof")
endif()

set(BASE ${CMAKE_CURRENT_LIST_DIR}/../../baselines/ref_c)
add_executable(mafc ${BASE}/pico_main.c ${CMAKE_CURRENT_LIST_DIR}/gen/mafc_${MODEL}.c)
target_include_directories(mafc PRIVATE ${BASE})
target_compile_definitions(mafc PRIVATE BUILD_NAME="${BUILD_NAME}")

pico_set_program_name(mafc "mafc")
pico_set_program_version(mafc "0.1")

pico_enable_stdio_uart(mafc 0)
pico_enable_stdio_usb(mafc 1)

target_link_libraries(mafc pico_stdlib hardware_clocks)

if (RUN_FROM_RAM)
    pico_set_binary_type(mafc copy_to_ram)
endif()
if (PROFILE_STAGES)
    target_compile_definitions(mafc PRIVATE REF_PROFILE=1)
endif()

pico_add_extra_outputs(mafc)
