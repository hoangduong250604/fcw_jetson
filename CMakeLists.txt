cmake_minimum_required(VERSION 3.10)
project(FCW_System LANGUAGES CXX)
link_libraries(stdc++fs)
# ============================================================================
# Forward Collision Warning System
# Vision-Based FCW Using YOLOv8 Detection & TTC Estimation on Jetson Nano
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")

# ============================================================================
# Platform Detection (Jetson Nano vs Desktop)
# ============================================================================
if(EXISTS "/etc/nv_tegra_release")
    set(PLATFORM_JETSON ON)
    message(STATUS "Platform: NVIDIA Jetson (Tegra)")
    add_definitions(-DPLATFORM_JETSON)
else()
    set(PLATFORM_JETSON OFF)
    message(STATUS "Platform: Desktop/x86")
    add_definitions(-DPLATFORM_DESKTOP)
endif()

# ============================================================================
# Compiler Flags
# ============================================================================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

if(PLATFORM_JETSON)
    # Jetson Nano specific: ARM + CUDA compute capability 5.3
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_53,code=sm_53")
else()
    # Desktop GPU (adjust compute capability as needed)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
endif()

# ============================================================================
# Find Required Packages
# ============================================================================

# ============================================================================
# Backend Options
# ============================================================================
if(PLATFORM_JETSON)
    option(USE_TENSORRT "Enable TensorRT backend (requires CUDA + TensorRT)" ON)
    option(USE_ONNXRUNTIME "Enable ONNX Runtime backend" OFF)
else()
    option(USE_TENSORRT "Enable TensorRT backend (requires CUDA + TensorRT)" OFF)
    option(USE_ONNXRUNTIME "Enable ONNX Runtime backend" ON)
endif()

# ============================================================================
# ONNX Runtime
# ============================================================================
if(USE_ONNXRUNTIME AND NOT USE_TENSORRT)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS ${ONNXRUNTIME_ROOT} "${CMAKE_SOURCE_DIR}/../onnxruntime"
        PATH_SUFFIXES lib include
    )
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
        HINTS ${ONNXRUNTIME_ROOT} "${CMAKE_SOURCE_DIR}/../onnxruntime"
        PATH_SUFFIXES lib
    )
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
        message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_INCLUDE_DIR}")
        message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIBRARY}")
        include_directories(${ONNXRUNTIME_INCLUDE_DIR})
    else()
        message(WARNING "ONNX Runtime not found, falling back to OpenCV DNN")
        set(USE_ONNXRUNTIME OFF)
    endif()
endif()

# ============================================================================
# TensorRT Backend
# ============================================================================
if(USE_TENSORRT)
    message(STATUS "TensorRT backend: ENABLED")

    # CUDA
    find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}
              /usr/local/cuda
              "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2"
        PATH_SUFFIXES include
    )
    find_library(CUDA_RUNTIME_LIBRARY NAMES cudart
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}
              /usr/local/cuda
              "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2"
        PATH_SUFFIXES lib lib64 lib/x64
    )
    if(NOT CUDA_RUNTIME_LIBRARY)
        find_library(CUDA_RUNTIME_LIBRARY NAMES cudart_static
            HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                  /usr/local/cuda
                  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2"
            PATH_SUFFIXES lib lib64 lib/x64
        )
    endif()

    if(CUDA_INCLUDE_DIR AND CUDA_RUNTIME_LIBRARY)
        message(STATUS "CUDA found: ${CUDA_INCLUDE_DIR}")
        message(STATUS "CUDA Runtime: ${CUDA_RUNTIME_LIBRARY}")
        include_directories(${CUDA_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "CUDA not found! Required when USE_TENSORRT=ON")
    endif()
else()
    if(USE_ONNXRUNTIME)
        message(STATUS "TensorRT backend: DISABLED (using ONNX Runtime)")
    else()
        message(STATUS "TensorRT backend: DISABLED (using OpenCV DNN)")
    endif()
endif()

# OpenCV (with CUDA support preferred)
find_package(OpenCV REQUIRED
    COMPONENTS core imgproc imgcodecs videoio highgui dnn video
)
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorRT (only when USE_TENSORRT=ON)
if(USE_TENSORRT)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
          /usr/include/aarch64-linux-gnu
          /usr/include/x86_64-linux-gnu
          /usr/include
    PATH_SUFFIXES include
)
# TensorRT libraries - support both Linux (nvinfer) and Windows (nvinfer_10) naming
find_library(TENSORRT_LIBRARY_INFER NAMES nvinfer nvinfer_10
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
          /usr/lib/aarch64-linux-gnu
          /usr/lib/x86_64-linux-gnu
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN NAMES nvinfer_plugin nvinfer_plugin_10
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
          /usr/lib/aarch64-linux-gnu
          /usr/lib/x86_64-linux-gnu
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)
find_library(TENSORRT_LIBRARY_PARSERS NAMES nvparsers nvparsers_10
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
          /usr/lib/aarch64-linux-gnu
          /usr/lib/x86_64-linux-gnu
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)
find_library(TENSORRT_LIBRARY_ONNX NAMES nvonnxparser nvonnxparser_10
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
          /usr/lib/aarch64-linux-gnu
          /usr/lib/x86_64-linux-gnu
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY_INFER)
    set(TENSORRT_FOUND ON)
    set(TENSORRT_LIBRARIES
        ${TENSORRT_LIBRARY_INFER}
        ${TENSORRT_LIBRARY_INFER_PLUGIN}
    )
    # Add optional parsers and ONNX libraries if found
    if(TENSORRT_LIBRARY_PARSERS)
        list(APPEND TENSORRT_LIBRARIES ${TENSORRT_LIBRARY_PARSERS})
    endif()
    if(TENSORRT_LIBRARY_ONNX)
        list(APPEND TENSORRT_LIBRARIES ${TENSORRT_LIBRARY_ONNX})
    endif()
    message(STATUS "TensorRT found: ${TENSORRT_INCLUDE_DIR}")
    include_directories(${TENSORRT_INCLUDE_DIR})
else()
    message(FATAL_ERROR "TensorRT not found! Set TENSORRT_ROOT to the installation path.")
endif()
endif()  # USE_TENSORRT

# yaml-cpp for config parsing (optional for now, needed for reading YAML configs)
find_package(yaml-cpp QUIET)
if(yaml-cpp_FOUND)
    message(STATUS "yaml-cpp found: ${YAML_CPP_INCLUDE_DIR}")
else()
    message(STATUS "yaml-cpp not found - YAML config parsing will be unavailable. Install via: vcpkg install yaml-cpp:x64-windows")
endif()

# Threads
find_package(Threads REQUIRED)

# ============================================================================
# Project Include Directories
# ============================================================================
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/camera
    ${CMAKE_SOURCE_DIR}/src/preprocessing
    ${CMAKE_SOURCE_DIR}/src/detection
    ${CMAKE_SOURCE_DIR}/src/tracking
    ${CMAKE_SOURCE_DIR}/src/distance
    ${CMAKE_SOURCE_DIR}/src/motion
    ${CMAKE_SOURCE_DIR}/src/ttc
    ${CMAKE_SOURCE_DIR}/src/risk
    ${CMAKE_SOURCE_DIR}/src/warning
    ${CMAKE_SOURCE_DIR}/src/visualization
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/gui
)

# ============================================================================
# Source Files
# ============================================================================
set(UTIL_SOURCES
    src/utils/logger.cpp
    src/utils/timer.cpp
)

# CUDA-MinGW compat only needed with TensorRT
if(USE_TENSORRT)
    list(APPEND UTIL_SOURCES src/utils/cuda_mingw_compat.cpp)
endif()

set(CAMERA_SOURCES
    src/camera/camera.cpp
)

set(PREPROCESSING_SOURCES
    src/preprocessing/image_preprocess.cpp
)

set(DETECTION_SOURCES
    src/detection/yolov8_detector.cpp
    src/detection/nms.cpp
)

set(TRACKING_SOURCES
    src/tracking/object_tracker.cpp
    src/tracking/track.cpp
    src/tracking/hungarian.cpp
)

set(DISTANCE_SOURCES
    src/distance/distance_estimator.cpp
    src/distance/bev_distance.cpp
)

set(MOTION_SOURCES
    src/motion/speed_estimator.cpp
    src/motion/kitti_oxts_reader.cpp
)

set(TTC_SOURCES
    src/ttc/ttc_calculator.cpp
)

set(RISK_SOURCES
    src/risk/collision_risk.cpp
)

set(WARNING_SOURCES
    src/warning/warning_system.cpp
    src/warning/audio_alert.cpp
)

set(VISUALIZATION_SOURCES
    src/visualization/visualization.cpp
)

set(CORE_SOURCES
    src/core/pipeline.cpp
    src/core/threaded_pipeline.cpp
)

set(GUI_SOURCES
    src/gui/gui_app.cpp
)

set(ALL_SOURCES
    ${UTIL_SOURCES}
    ${CAMERA_SOURCES}
    ${PREPROCESSING_SOURCES}
    ${DETECTION_SOURCES}
    ${TRACKING_SOURCES}
    ${DISTANCE_SOURCES}
    ${MOTION_SOURCES}
    ${TTC_SOURCES}
    ${RISK_SOURCES}
    ${WARNING_SOURCES}
    ${VISUALIZATION_SOURCES}
    ${CORE_SOURCES}
    ${GUI_SOURCES}
)

# ============================================================================
# Build Library
# ============================================================================
add_library(fcw_lib STATIC ${ALL_SOURCES})

# Link libraries
target_link_libraries(fcw_lib
    ${OpenCV_LIBS}
    Threads::Threads
    stdc++fs
)

# Link TensorRT + CUDA when enabled
if(USE_TENSORRT)
    target_link_libraries(fcw_lib ${TENSORRT_LIBRARIES} ${CUDA_RUNTIME_LIBRARY})
    target_compile_definitions(fcw_lib PUBLIC USE_TENSORRT=1)
elseif(USE_ONNXRUNTIME)
    target_link_libraries(fcw_lib ${ONNXRUNTIME_LIBRARY})
    target_compile_definitions(fcw_lib PUBLIC USE_ONNXRUNTIME=1)
endif()

# Add Windows-specific libraries
if(WIN32)
    target_link_libraries(fcw_lib shlwapi)
endif()

# Conditionally link yaml-cpp if found
if(yaml-cpp_FOUND)
    target_link_libraries(fcw_lib yaml-cpp)
    target_compile_definitions(fcw_lib PUBLIC HAVE_YAML_CPP=1)
endif()

# ============================================================================
# Main Executable
# ============================================================================
add_executable(fcw_system src/main.cpp)
target_link_libraries(fcw_system fcw_lib)

# ============================================================================
# Install
# ============================================================================
install(TARGETS fcw_system DESTINATION bin)
install(DIRECTORY config/ DESTINATION etc/fcw-system)
install(DIRECTORY models/ DESTINATION share/fcw-system/models)

# ============================================================================
# Custom Targets
# ============================================================================
# Model conversion helper
add_custom_target(convert_model
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/convert_model.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Converting ONNX model to TensorRT engine..."
)

# Run demo
add_custom_target(run_demo
    COMMAND $<TARGET_FILE:fcw_system>
        --config ${CMAKE_SOURCE_DIR}/config/system_config.yaml
    DEPENDS fcw_system
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running FCW System demo..."
)

message(STATUS "")
message(STATUS "========================================")
message(STATUS "  FCW System Configuration Summary")
message(STATUS "========================================")
message(STATUS "  Platform:    ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Jetson:      ${PLATFORM_JETSON}")
message(STATUS "  Build Type:  ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA:        ${CUDA_VERSION}")
message(STATUS "  OpenCV:      ${OpenCV_VERSION}")
message(STATUS "  TensorRT:    ${TENSORRT_INCLUDE_DIR}")
message(STATUS "  OnnxRuntime: ${USE_ONNXRUNTIME}")
message(STATUS "========================================")
