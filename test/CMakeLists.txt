function(ut name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} speedup_static)
endfunction(ut)

ut(test_conv)