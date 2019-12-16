function(ut name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} speedup_static)
endfunction(ut)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static")

ut(test_conv)
ut(test_blk)