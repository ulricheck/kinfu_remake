include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../kfusion/include/)

# Not recommended by CMake to use GLOB
# file(GLOB srcs *.cpp *.hpp)

# APP 01: OpenNI capture
if (OPENNI_FOUND)
    add_executable(kinfu demo.cpp)
    target_link_libraries(kinfu ${OpenCV_LIBS} kfusion)

    set_target_properties(kinfu PROPERTIES
        DEBUG_POSTFIX "d"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    install(TARGETS kinfu RUNTIME DESTINATION bin COMPONENT main)
    install(FILES demo.cpp DESTINATION app COMPONENT main)
endif(OPENNI_FOUND)

# APP 02: BinGrabber capture
add_executable(kinfu_bin demo_bin.cpp)
target_link_libraries(kinfu_bin ${OpenCV_LIBS} ${PCL_LIBRARIES} kfusion)

set_target_properties(kinfu_bin PROPERTIES
        DEBUG_POSTFIX "d"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

install(TARGETS kinfu_bin RUNTIME DESTINATION bin COMPONENT main)
install(FILES demo_bin.cpp DESTINATION app COMPONENT main)

# APP 03: Ubitrack capture
if (UBITRACK_FOUND)
    add_executable(kinfu_ubitrack demo_ubitrack.cpp)
    target_link_libraries(kinfu_ubitrack ${OpenCV_LIBS} ${UBITRACK_LIBRARIES} kfusion)

    set_target_properties(kinfu_ubitrack PROPERTIES
        DEBUG_POSTFIX "d"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    install(TARGETS kinfu_ubitrack RUNTIME DESTINATION bin COMPONENT main)
    install(FILES demo_ubitrack.cpp DESTINATION app COMPONENT main)
endif(UBITRACK_FOUND)
