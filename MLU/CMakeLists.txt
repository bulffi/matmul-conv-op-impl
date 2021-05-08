add_custom_command(
    OUTPUT 
        ${CMAKE_CURRENT_SOURCE_DIR}/libmlu.so
        ${CMAKE_CURRENT_SOURCE_DIR}/__header.h
    COMMAND "make"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(
    my_custom_target_that_always_runs ALL
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/__header.h
)

add_library(
    mlu_op
    SHARED
    mlu_op_pybind_wrapper.cpp
)

add_dependencies(
    mlu_op
    my_custom_target_that_always_runs
)

set_target_properties(
    mlu_op
    PROPERTIES 
        PREFIX ""
)

target_link_libraries(
    mlu_op
    ${CMAKE_CURRENT_SOURCE_DIR}/libmlu.so pybind11::module
)