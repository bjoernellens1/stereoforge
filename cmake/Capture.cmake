# Private upstream build: no global compiler options, no SDK/CUDA dependency.
include(FetchContent)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir OUTPUT_VARIABLE SF_PYBIND_DIR OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE SF_PYBIND_RESULT)
if(NOT SF_PYBIND_RESULT EQUAL 0)
  message(FATAL_ERROR "Capture needs pybind11==2.13.6 in the selected Python environment")
endif()
find_package(pybind11 2.13.6 CONFIG REQUIRED PATHS "${SF_PYBIND_DIR}" NO_DEFAULT_PATH)
include("${CMAKE_CURRENT_LIST_DIR}/ZedCapture.cmake")
# The pinned video module uses Linux V4L2 directly; it does not include/use libusb.
pybind11_add_module(_zed_capture "${CMAKE_CURRENT_LIST_DIR}/../src/zed_capture.cpp")
target_link_libraries(_zed_capture PRIVATE stereoforge_zed_private)
target_compile_features(_zed_capture PRIVATE cxx_std_20)
set_target_properties(_zed_capture PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/native")

# Preserve full license notices beside the optional native capture binary.
set(SF_CAPTURE_LICENSE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/licenses")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/native/licenses")
foreach(name zed-open-capture-MIT.txt hidapi-BSD-3-Clause.txt pybind11-BSD-3-Clause.txt)
  configure_file("${SF_CAPTURE_LICENSE_SOURCE_DIR}/${name}" "${CMAKE_BINARY_DIR}/native/licenses/${name}" COPYONLY)
endforeach()
