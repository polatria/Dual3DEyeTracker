cmake_minimum_required(VERSION 2.8)
SET(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "limited configs" FORCE)

project( EyeDetector )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
set(CMAKE_INSTALL_PREFIX "./install")

# Your custom paths to Boost and OpenCV libraries
set(BOOST_ROOT "C:/SDK/boost_1_64_0")
set(OpenCV_DIR "C:/SDK/CV/3.4.6/install")

# Bundled external libraries
set(CMAKE_PREFIX_PATH  "${PROJECT_SOURCE_DIR}/external/Ceres-1.11/CMake")
set(SPII_INSTALL_DIR   "${PROJECT_SOURCE_DIR}/external/spii-3.0.0")
set(EIGEN_INCLUDE_DIR  "${PROJECT_SOURCE_DIR}/external/Eigen-3.2.7")
set(EIGEN3_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/Eigen-3.2.7")

# Setup Boost path
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
unset(Boost_INCLUDE_DIR CACHE)
unset(Boost_LIBRARY_DIRS CACHE)
set(BOOST_LIBRARYDIR ${BOOST_ROOT}/stage/x64/lib)


find_package ( Boost 1.62 COMPONENTS filesystem system timer thread date_time chrono regex serialization REQUIRED )
find_package ( OpenCV REQUIRED )
find_package ( Eigen3 REQUIRED )
find_package ( Ceres REQUIRED )
find_package ( PythonLibs REQUIRED )
find_package ( Python3 COMPONENTS NumPy )

# Find spii
set(spii_INCLUDE_DIRS ${SPII_INSTALL_DIR}/include)
find_library (spii_LIBRARY_RELEASE
              spii
              PATHS ${SPII_INSTALL_DIR}/lib/ ${SPII_INSTALL_DIR}/lib/Release)
find_library (meschach_LIBRARY_RELEASE
              meschach
              PATHS ${SPII_INSTALL_DIR}/lib/ ${SPII_INSTALL_DIR}/lib/Release)
if (spii_LIBRARY_RELEASE AND meschach_LIBRARY_RELEASE)
    set(spii_LIBRARIES ${spii_LIBRARY_RELEASE} ${meschach_LIBRARY_RELEASE})
else()
    set(spii_LIBRARIES "")
endif()
message(STATUS "spii_LIBRARIES:\"${spii_LIBRARIES}\"")
# copy dll
file(COPY ${SPII_INSTALL_DIR}/bin/spii.dll DESTINATION ${PROJECT_BINARY_DIR}/main)

# copy OpenCV dlls
file(GLOB OPENCV_DLLS
  "${OpenCV_DIR}/x64/vc15/bin/*.dll"
)
file(COPY ${OPENCV_DLLS} DESTINATION ${PROJECT_BINARY_DIR}/main)

message(STATUS "")
message(STATUS "--- Include directories ---" )
message(STATUS " OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}" )
message(STATUS " Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}" )
message(STATUS "---------------------------" )
message(STATUS "")

add_subdirectory (EyeDetector)
add_subdirectory (singleeyefitter)
