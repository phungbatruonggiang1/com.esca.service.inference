cmake_minimum_required(VERSION 3.16)
project(esca-inference VERSION 0.1.0 LANGUAGES CXX)

# The AI inference service for the ESCA board:
#   libesca-inference           the engine: one model on one backend (TFLite)
#   libesca-resource-allocator  which models are loaded, where each request runs
#   esca-inferenced             the D-Bus service, vn.sparc.esca.Inference1
#   libesca-inference-client    what applications link
#   esca-inferctl               the service from a terminal

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
find_package(nlohmann_json 3 REQUIRED)
pkg_check_modules(SDBUSCPP REQUIRED IMPORTED_TARGET sdbus-c++)

# TensorFlow Lite installs no CMake package; see esca-vision for why the two
# Abseil logging libraries are named explicitly.
find_library(TFLITE_LIB NAMES tensorflow-lite REQUIRED)
find_path(TFLITE_INCLUDE_DIR tensorflow/lite/interpreter.h REQUIRED)
find_library(ABSL_LOG_MESSAGE_LIB NAMES absl_log_internal_message REQUIRED)
find_library(ABSL_LOG_NULLGUARD_LIB NAMES absl_log_internal_nullguard REQUIRED)

set(PUBLIC_INCLUDE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# --- libesca-inference --------------------------------------------------------
add_library(esca-inference SHARED
    src/common/types.cpp
    src/common/model_config.cpp
    src/common/logger.cpp
    src/inference/tflite_strategy.cpp)
target_include_directories(esca-inference
    PUBLIC ${PUBLIC_INCLUDE}
    PRIVATE src/common src/inference ${TFLITE_INCLUDE_DIR})
target_link_libraries(esca-inference
    PRIVATE nlohmann_json::nlohmann_json ${TFLITE_LIB} ${ABSL_LOG_MESSAGE_LIB} ${ABSL_LOG_NULLGUARD_LIB}
    PUBLIC Threads::Threads)
set_target_properties(esca-inference PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 0)

# --- libesca-resource-allocator ------------------------------------------------
add_library(esca-resource-allocator SHARED
    src/allocator/device_selector.cpp
    src/allocator/memory_monitor.cpp
    src/allocator/model.cpp
    src/allocator/model_manager.cpp
    src/allocator/policies.cpp
    src/allocator/scheduler.cpp)
target_include_directories(esca-resource-allocator
    PUBLIC ${PUBLIC_INCLUDE}
    PRIVATE src/common src/inference src/allocator)
target_link_libraries(esca-resource-allocator PUBLIC esca-inference Threads::Threads)
set_target_properties(esca-resource-allocator PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 0)

# --- esca-inferenced -------------------------------------------------------------
add_executable(esca-inferenced
    src/service/main.cpp
    src/service/service.cpp
    src/service/config_manager.cpp
    src/service/buffer_allocator.cpp
    src/service/context_manager.cpp
    src/service/model_registration.cpp)
target_include_directories(esca-inferenced PRIVATE src/common src/inference src/allocator src/service)
target_link_libraries(esca-inferenced PRIVATE
    esca-resource-allocator nlohmann_json::nlohmann_json PkgConfig::SDBUSCPP rt)

# --- libesca-inference-client ------------------------------------------------------
add_library(esca-inference-client SHARED
    src/client/service_client.cpp
    src/client/sessions.cpp)
target_include_directories(esca-inference-client
    PUBLIC ${PUBLIC_INCLUDE}
    PRIVATE src/common src/inference)
target_link_libraries(esca-inference-client
    PUBLIC esca-inference
    PRIVATE PkgConfig::SDBUSCPP nlohmann_json::nlohmann_json rt)
set_target_properties(esca-inference-client PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 0)

# --- esca-inferctl -------------------------------------------------------------------
add_executable(esca-inferctl src/tools/inferctl.cpp)
target_link_libraries(esca-inferctl PRIVATE esca-inference-client)

# --- tests: the allocator against fakes, runnable on the board ---------------------
add_executable(esca-inference-allocator-test test/allocator_test.cpp)
target_include_directories(esca-inference-allocator-test PRIVATE src/common src/inference src/allocator test)
target_link_libraries(esca-inference-allocator-test PRIVATE esca-resource-allocator nlohmann_json::nlohmann_json)

install(TARGETS esca-inference esca-resource-allocator esca-inference-client
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS esca-inferenced RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR})
install(TARGETS esca-inferctl RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS esca-inference-allocator-test RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/esca-inference)
install(DIRECTORY include/esca DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES data/vn.sparc.esca.Inference1.conf DESTINATION ${CMAKE_INSTALL_DATADIR}/dbus-1/system.d)
install(FILES data/esca-inferenced.service DESTINATION ${CMAKE_INSTALL_LIBDIR}/systemd/system)
install(FILES data/inference.conf DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/esca)
install(DIRECTORY data/models.d/ DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/esca/inference/models.d)
