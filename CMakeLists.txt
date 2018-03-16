cmake_minimum_required(VERSION 2.8)
project(install-project)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
file(GLOB SOURCES *.cpp)

set(EXECUTABLE caffe2_inference_test)


# add modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules")

message(${CMAKE_MODULE_PATH})


# find packages
find_package(Caffe2 REQUIRED)	
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)


if(USE_MKL)
	find_package(MKL REQUIRED)
endif()

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_ARCH -gencode arch=compute_50,code=sm_50)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -g -G --std=c++11)


set(PROJECT_LIBRARIES ${Caffe2_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY} ${MKL_LIBRARIES} )		
set(PROJECT_INCLUDES ${Caffe2_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${MKL_INCLUDE_DIR} )
set(LIBS -ldl make)

message(Project_include_path: ${PROJECT_INCLUDES})

include_directories(${PROJECT_INCLUDES} ${EIGEN3_INCLUDE_DIR} /usr/local/cuda/include )
CUDA_ADD_EXECUTABLE(${EXECUTABLE}  ${SOURCES})

target_link_libraries(${EXECUTABLE} ${PROJECT_LIBRARIES} -L/usr/local/cuda/lib64 -lcuda -lcudart -lprotobuf -lglog -lgflags -ldl -lrt -L/usr/local/lib/)



