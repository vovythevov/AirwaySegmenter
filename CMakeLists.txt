project(AirwaySegmenter)
cmake_minimum_required(VERSION 2.8)

#Find packages
find_package(SlicerExecutionModel REQUIRED)
include(${SlicerExecutionModel_USE_FILE})

find_package(VTK REQUIRED)
INCLUDE(${VTK_USE_FILE})

find_package(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

IF (ITK_VERSION_MAJOR VERSION_LESS "4")
  FATAL_ERROR("ITK version must be at least 4.0 ")
ENDIF()

#Testing
OPTION(BUILD_TESTING "Build tests." ON)

IF(BUILD_TESTING)
  INCLUDE(CTest)
  SET(TEMP "${CMAKE_BINARY_DIR}/Temp")
ENDIF()

#Compatibility between slicer and the executable
SET(SLICER_ITK_DIFFERENT true CACHE BOOL "Check if compiling with a different ITK than slicer")
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/AirwaySegmenterConfig.h.in
               ${CMAKE_BINARY_DIR}/AirwaySegmenterConfig.h)

include_directories(Code)

#Finally the project
SEMMacroBuildCLI(
  NAME ${PROJECT_NAME}
  TARGET_LIBRARIES ${ITK_LIBRARIES} ${VTK_LIBRAIRIES}
  EXECUTABLE_ONLY
  )
  