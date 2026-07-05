cmake_minimum_required (VERSION 3.8)

add_executable (vits-test "vits.cpp")

target_include_directories(vits-test PUBLIC . ../thirdparty )

target_link_libraries(vits-test PUBLIC vits)
target_link_libraries(vits-test PRIVATE wave)