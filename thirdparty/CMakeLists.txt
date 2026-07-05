set(Z_TARGET zip)
add_library(${Z_TARGET} OBJECT zip.c zip.h miniz.h)

set(CJSON_TARGET cjson)
add_library(${CJSON_TARGET} OBJECT cJSON.cpp cJSON.h)

set(WAVE_TARGET wave)
add_library(${WAVE_TARGET} OBJECT wave.cpp wave.h)

target_include_directories(${Z_TARGET} PUBLIC .)
target_include_directories(${CJSON_TARGET} PUBLIC .)
target_include_directories(${WAVE_TARGET} PUBLIC .)