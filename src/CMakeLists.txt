set(SRC 
    ${SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/sgemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/conv_im2col.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/conv_wino.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/conv.cpp
    )

add_library(speedup_shared SHARED ${SRC})
add_library(speedup_static STATIC ${SRC})

set_target_properties(speedup_shared PROPERTIES OUTPUT_NAME "speedup")
set_target_properties(speedup_static PROPERTIES OUTPUT_NAME "speedup")