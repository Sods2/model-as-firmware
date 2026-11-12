# Bring-up firmware: LED blink + USB serial + DWT cycle-counter timing

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

include(pico_sdk_import.cmake)

project(hello C CXX ASM)

pico_sdk_init()

add_executable(hello hello.c)

pico_set_program_name(hello "hello")
pico_set_program_version(hello "0.1")

pico_enable_stdio_uart(hello 0)
pico_enable_stdio_usb(hello 1)

target_link_libraries(hello pico_stdlib hardware_clocks)

pico_add_extra_outputs(hello)
