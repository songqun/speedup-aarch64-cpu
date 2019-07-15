aux_source_directory(. SRC)

add_library(speedup_shared SHARED ${SRC})
add_library(speedup_static STATIC ${SRC})

set_target_properties(speedup_shared PROPERTIES OUTPUT_NAME "speedup")
set_target_properties(speedup_static PROPERTIES OUTPUT_NAME "speedup")