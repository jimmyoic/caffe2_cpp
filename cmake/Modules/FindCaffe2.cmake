





include(FindPackageHandleStandardArgs)
unset(CAFFE2_FOUND)

find_library(Caffe2_LIBRARY_GPU 
	NAMES caffe2_gpu
	HINTS /usr/local/lib
)


find_library(Caffe2_LIBRARY_CPU 
		NAMES caffe2
        HINTS /usr/local/lib
)




# set Caffe2_FOUND
find_package_handle_standard_args(Caffe2 DEFAULT_MSG Caffe2_LIBRARY_CPU Caffe2_LIBRARY_GPU)
# set external variables for usage in CMakeLists.txt
if(CAFFE2_FOUND)
    set(Caffe2_LIBRARY ${Caffe2_LIBRARY_CPU} ${Caffe2_LIBRARY_GPU})
    set(Caffe2_INCLUDE_DIR "/usr/local/include")
endif()

# hide locals from GUI
mark_as_advanced(Caffe2_INCLUDE_DIR Caffe2_LIBRARY)
