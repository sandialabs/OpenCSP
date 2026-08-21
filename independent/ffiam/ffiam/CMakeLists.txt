# CMakeList.txt : CMake project for ffiam
# CUDA-accelerated irradiance analysis for concentrating solar power plants.

cmake_minimum_required(VERSION 3.21)

# Enable Hot Reload for MSVC compilers if supported.
if(POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# Record whether the caller chose the CUDA architectures. This MUST be sampled
# before project(), because project() unconditionally defines
# CMAKE_CUDA_ARCHITECTURES with a default (52 -- Maxwell), after which "did the
# user ask for this?" is no longer answerable.
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(FFIAM_CUDA_ARCHS_FROM_USER TRUE)
else()
    set(FFIAM_CUDA_ARCHS_FROM_USER FALSE)
endif()

project(ffiam LANGUAGES C CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Set the default build to Release.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif()

# CMake module path for custom scripts
set(CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/tools/CMake"
    ${CMAKE_MODULE_PATH}
)

# Output directories - all targets output to same location
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")


##############################################################################
# Include Directories (shared by all targets)
##############################################################################

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/fmt/include
    ${CMAKE_SOURCE_DIR}/external/solpos
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)


##############################################################################
# CUDA Configuration
##############################################################################

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # For Xcode/clang compatibility
    if(USING_CLANG_C)
        set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
    endif()

    # Windows-specific NVCC flags
    if(USING_WINDOWS_CL)
        list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
    endif()

    # macOS rpath for CUDA toolkit
    if(APPLE)
        set(CUDA_TOOLKIT_RPATH_FLAG "-Wl,-rpath,${CUDA_TOOLKIT_ROOT_DIR}/lib")
    endif()
else()
    message(FATAL_ERROR "CUDA compiler not found. CUDA is required to build FFIAM.")
endif()

# Decide which CUDA architectures to build for.
#
# If the caller passed -DCMAKE_CUDA_ARCHITECTURES=..., honour it verbatim.
# Otherwise pick a sane default -- NOT CMake's, which is 52 (Maxwell, 2014) and
# silently leaves modern GPUs running JIT-compiled PTX.
#
# "native" builds only for the GPU in this machine, but CMake only understands
# it from 3.24 (older versions pass it through unexpanded and nvcc dies with
# "Unsupported gpu architecture 'compute_'"), and it needs a GPU visible at
# configure time. Fall back to an explicit list, trimmed to what the detected
# nvcc actually supports: sm_89 needs CUDA >= 11.8, sm_120 needs CUDA >= 12.8.
if(FFIAM_CUDA_ARCHS_FROM_USER)
    set(FFIAM_CUDA_ARCHS ${CMAKE_CUDA_ARCHITECTURES})
    message(STATUS "FFIAM: using caller-supplied CUDA architectures: ${FFIAM_CUDA_ARCHS}")
elseif(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(FFIAM_CUDA_ARCHS native)
    message(STATUS "FFIAM: CUDA architectures = native (this machine's GPU)")
else()
    set(FFIAM_CUDA_ARCHS 75)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
        list(APPEND FFIAM_CUDA_ARCHS 86 89)
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
        list(APPEND FFIAM_CUDA_ARCHS 120)
    endif()
    message(STATUS "FFIAM: CMake ${CMAKE_VERSION} predates 'native' support (3.24); "
                   "CUDA architectures = ${FFIAM_CUDA_ARCHS} "
                   "(pass -DCMAKE_CUDA_ARCHITECTURES=... to override)")
endif()

# Helper function to configure CUDA targets
function(configure_cuda_target target_name)
    target_link_libraries(${target_name} CUDA::cudart CUDA::cuda_driver)
    set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES ${FFIAM_CUDA_ARCHS})
endfunction()


##############################################################################
# OpenMP Configuration (optional; CPU backend uses it when available)
##############################################################################

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - CPU backend will parallelize heliostat loop")
    set(FFIAM_HAVE_OPENMP TRUE)
else()
    message(WARNING "OpenMP not found - CPU backend will run single-threaded")
    set(FFIAM_HAVE_OPENMP FALSE)
endif()

function(configure_openmp target_name)
    if(FFIAM_HAVE_OPENMP)
        target_link_libraries(${target_name} OpenMP::OpenMP_CXX)
        target_compile_definitions(${target_name} PRIVATE FFIAM_HAVE_OPENMP)
    endif()
endfunction()


##############################################################################
# Source Files (defined once, shared by targets)
##############################################################################

set(FFIAM_SOURCES
    src/main/ffiam.cu
    src/compute/irradiance_cpu.cpp
    src/compute/irradiance_cuda.cu
    src/solar/solar_position.cpp
    src/core/heliostat.cpp
    src/core/voxel.cpp
    src/api/preset_sites.cpp
    src/memory/pool.c
    src/io/csv_reader.cpp
    external/fmt/src/format.cc
)

# Headers listed for IDE visibility (not required for compilation)
set(FFIAM_HEADERS
    include/ffiam/ffiam.h
    include/ffiam/types.h
    include/ffiam/export.h
    src/util/cuda_utils.h
    src/io/csv_reader.h
)

set(FFIAM_TEST_SOURCES
    tests/test_main.cpp
    tests/test_coordinate_conv.cpp
    tests/test_voxel_indexing.cpp
    tests/test_solar_position.cpp
    tests/test_preset_sites.cpp
    tests/test_aim_fixed_normal.cpp
    tests/test_aim_strategies.cpp
    tests/test_ambient_irradiance.cpp
    tests/test_boundary_conditions.cpp
    tests/test_cpu_irradiance.cpp
    tests/test_csv_reader.cpp
    tests/test_math_primitives.cpp
)

# Test target needs subset of sources (no CUDA entry point)
set(FFIAM_TEST_LIB_SOURCES
    src/core/voxel.cpp
    src/solar/solar_position.cpp
    src/core/heliostat.cpp
    src/api/preset_sites.cpp
    src/memory/pool.c
    src/io/csv_reader.cpp
    src/compute/irradiance_cpu.cpp
    external/fmt/src/format.cc
)


##############################################################################
# Build Targets
##############################################################################

# ==== EXECUTABLE ====
# Standalone executable for testing and development
add_executable(ffiam ${FFIAM_SOURCES} ${FFIAM_HEADERS})
target_include_directories(ffiam PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
configure_cuda_target(ffiam)
configure_openmp(ffiam)

add_custom_command(TARGET ffiam POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/data $<TARGET_FILE_DIR:ffiam>/data)


# ==== SHARED LIBRARY (DLL) ====
# For Python consumption via ctypes
if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
add_library(ffiam_lib SHARED ${FFIAM_SOURCES} ${FFIAM_HEADERS})
target_include_directories(ffiam_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
configure_cuda_target(ffiam_lib)
configure_openmp(ffiam_lib)

add_custom_command(TARGET ffiam_lib POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/data $<TARGET_FILE_DIR:ffiam>/data)

# Copy shared library to pyffiam for Python integration
add_custom_command(TARGET ffiam_lib POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:ffiam_lib>
        ${CMAKE_SOURCE_DIR}/../pyffiam/src/pyffiam/external/$<TARGET_FILE_NAME:ffiam_lib>)


# ==== SHARED LIBRARY (CPU-only) ====
# For machines without CUDA. Same source content, FFIAM_CPU_ONLY defined so the
# CUDA bodies of .cu files are #ifdef'd out, leaving only host-side C++. We
# can't reuse the .cu files directly because LANGUAGE is a source-file property
# (not target-scoped) — overriding it would also affect ffiam_lib. Instead we
# copy each .cu into the build tree with a .cpp extension and feed those copies
# only to this target.
set(FFIAM_CPU_SOURCES "")
foreach(src ${FFIAM_SOURCES})
    if(src MATCHES "\\.cu$")
        get_filename_component(src_name ${src} NAME_WE)
        set(cpp_copy "${CMAKE_CURRENT_BINARY_DIR}/cpu_sources/${src_name}.cpp")
        add_custom_command(
            OUTPUT ${cpp_copy}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/${src} ${cpp_copy}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src}
            COMMENT "Copying ${src} -> ${cpp_copy} for CPU build"
        )
        list(APPEND FFIAM_CPU_SOURCES ${cpp_copy})
    else()
        list(APPEND FFIAM_CPU_SOURCES ${src})
    endif()
endforeach()

add_library(ffiam_lib_cpu SHARED ${FFIAM_CPU_SOURCES} ${FFIAM_HEADERS})
target_include_directories(ffiam_lib_cpu PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_compile_definitions(ffiam_lib_cpu PRIVATE FFIAM_CPU_ONLY)
configure_openmp(ffiam_lib_cpu)

add_custom_command(TARGET ffiam_lib_cpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/data $<TARGET_FILE_DIR:ffiam_lib_cpu>/data)

# Copy CPU shared library to pyffiam for Python integration
add_custom_command(TARGET ffiam_lib_cpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:ffiam_lib_cpu>
        ${CMAKE_SOURCE_DIR}/../pyffiam/src/pyffiam/external/$<TARGET_FILE_NAME:ffiam_lib_cpu>)


# ==== STATIC LIBRARY ====
# For Unreal Engine integration (must be built in Release config)
add_library(ffiam_ue STATIC ${FFIAM_SOURCES} ${FFIAM_HEADERS})
target_include_directories(ffiam_ue PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
configure_cuda_target(ffiam_ue)
configure_openmp(ffiam_ue)

add_custom_command(TARGET ffiam_ue POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/data $<TARGET_FILE_DIR:ffiam>/data)


##############################################################################
# Unit Tests
##############################################################################

option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_CUDA_TESTS "Build CUDA-gated kernel tests (requires CUDA)" ON)

if(BUILD_TESTS)
    set(FFIAM_TEST_ALL_SOURCES ${FFIAM_TEST_SOURCES} ${FFIAM_TEST_LIB_SOURCES})

    # Optional CUDA-gated kernel tests: directly launch Cuda_ComputeHeliostatIrradiance
    # and assert parity with the CPU backend. Requires CUDA to be available and
    # links the CUDA irradiance translation unit into the tests target.
    if(BUILD_CUDA_TESTS AND CMAKE_CUDA_COMPILER)
        list(APPEND FFIAM_TEST_ALL_SOURCES
            tests/test_cuda_irradiance.cu
            tests/test_field_analysis.cpp
            src/compute/irradiance_cuda.cu
            src/main/ffiam.cu
        )
    endif()

    add_executable(ffiam_tests ${FFIAM_TEST_ALL_SOURCES})

    target_include_directories(ffiam_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/external/fmt/include
        ${CMAKE_SOURCE_DIR}/external/solpos
        ${CMAKE_SOURCE_DIR}/tests
    )

    configure_openmp(ffiam_tests)

    if(BUILD_CUDA_TESTS AND CMAKE_CUDA_COMPILER)
        configure_cuda_target(ffiam_tests)
        # Suppress the standalone main() in ffiam.cu so test_main.cpp owns the
        # entry point for ffiam_tests.
        target_compile_definitions(ffiam_tests PRIVATE FFIAM_NO_MAIN)
    endif()

    enable_testing()
    add_test(NAME ffiam_unit_tests COMMAND ffiam_tests)
endif()


##############################################################################
# IDE Organization
##############################################################################

source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")
source_group("Header Files" REGULAR_EXPRESSION ".+\\.h$")
