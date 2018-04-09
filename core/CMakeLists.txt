# CMake file for compiling the C++ core library
# library sources
set(core_sources   dtss_client.cpp experimental.cpp dream_optimizer.cpp sceua_optimizer.cpp
    time_series_dd.cpp core_serialization.cpp time_series_serialization.cpp
    utctime_utilities.cpp expression_serialization.cpp
)
add_library(shyft_core STATIC ${core_sources} )

install(TARGETS shyft_core ARCHIVE DESTINATION lib COMPONENT shyft_core EXCLUDE_FROM_ALL)
install(FILES ${CMAKE_SOURCE_DIR}/build_support/shyft_core-config.cmake DESTINATION lib/cmake/shyft_core COMPONENT shyft_core EXCLUDE_FROM_ALL)
install(DIRECTORY  ${CMAKE_SOURCE_DIR}/core  DESTINATION include/shyft  COMPONENT shyft_core EXCLUDE_FROM_ALL FILES_MATCHING PATTERN "*.h" PATTERN "obj" EXCLUDE)
