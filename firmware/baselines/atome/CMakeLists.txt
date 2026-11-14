# Atome LM baseline on the Pico 2 (Phase 3): the upstream Atome C99 engine, unmodified (upstream/), with a
# serial harness (pico_main.c). The .atome blob is linked in with .incbin (model_blob.S).
#
#   -DATOME_BIN=<path>      model blob (default: Atome's 944K checkpoint, train/atome-lm/checkpoints/atome_944k.bin)
#   -DMAX_SEQ=24            engine window (ATOME_MAX_SEQ); the ESP32-WROOM demo uses 24
#   -DRUN_FROM_RAM=ON       copy everything (code + weights) to SRAM at boot
#   -DATOME_D_MODEL=... -DATOME_N_LAYERS=... -DATOME_D_HEAD=...   must match the blob

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
set(ATOME_BIN ${CMAKE_CURRENT_LIST_DIR}/../../../train/atome-lm/checkpoints/atome_944k.bin CACHE FILEPATH "Model blob")
set(MAX_SEQ 24 CACHE STRING "ATOME_MAX_SEQ")
set(ATOME_D_MODEL 256 CACHE STRING "")
set(ATOME_N_LAYERS 8 CACHE STRING "")
set(ATOME_D_HEAD 64 CACHE STRING "")

include(${CMAKE_CURRENT_LIST_DIR}/../../rp2350/hello/pico_sdk_import.cmake)

project(atome C CXX ASM)

pico_sdk_init()

option(RUN_FROM_RAM "Copy the whole program (code and weights) to SRAM at boot" OFF)
get_filename_component(BLOB_NAME ${ATOME_BIN} NAME_WE)
set(BUILD_NAME "atome_${BLOB_NAME}_s${MAX_SEQ}")
if (RUN_FROM_RAM)
    string(APPEND BUILD_NAME "_ram")
endif()

add_executable(atome pico_main.c upstream/atome.c model_blob.S)
set_source_files_properties(model_blob.S PROPERTIES OBJECT_DEPENDS ${ATOME_BIN})

target_include_directories(atome PRIVATE ${CMAKE_CURRENT_LIST_DIR}/upstream)
target_compile_definitions(atome PRIVATE
    ATOME_D_MODEL=${ATOME_D_MODEL}
    ATOME_N_LAYERS=${ATOME_N_LAYERS}
    ATOME_D_HEAD=${ATOME_D_HEAD}
    ATOME_MAX_SEQ=${MAX_SEQ}
    ATOME_N_PATHWAYS=3
    ATOME_VOCAB_SIZE=256
    ATOME_KERNEL_SIZE=5
    ATOME_TOP_K=4
    ATOME_BIN_PATH="${ATOME_BIN}"
    BUILD_NAME="${BUILD_NAME}")

pico_set_program_name(atome "atome")
pico_enable_stdio_uart(atome 0)
pico_enable_stdio_usb(atome 1)

target_link_libraries(atome pico_stdlib hardware_clocks)

if (RUN_FROM_RAM)
    pico_set_binary_type(atome copy_to_ram)
endif()

pico_add_extra_outputs(atome)
