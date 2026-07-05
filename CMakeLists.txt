cmake_minimum_required(VERSION 3.8)
project("vits")

if (WIN32)
	add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

	if (BUILD_SHARED_LIBS)
		set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
	endif()
endif()

set(BUILD_SHARED_LIBS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(thirdparty)
add_subdirectory(test)
add_subdirectory(ggml)

add_library(vits 
	"vits.cpp" 
	"include/vits.h"
	"ggml_extend.hpp" 
	"attentions.hpp" 
	"modules.hpp" 
	"models.hpp" 
	"util.cpp" 
	"util.h" 
	"model.cpp" 
	"model.h" 
	"transforms.hpp" 
	"vits.hpp"
				)

target_include_directories(vits PUBLIC . ./include )
target_compile_features   (vits PUBLIC cxx_std_11)

target_link_libraries(vits PUBLIC ggml)
target_link_libraries(vits PRIVATE zip cjson)

if (BUILD_SHARED_LIBS)
	set_target_properties(vits PROPERTIES POSITION_INDEPENDENT_CODE ON)
	target_compile_definitions(vits PRIVATE VITS_SHARED VITS_BUILD)
else()
	target_compile_definitions(vits PRIVATE VITS_BUILD)
endif()