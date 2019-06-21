include_directories (
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${spii_INCLUDE_DIRS}
  "../singleeyefitter"
  ${PYTHON_INCLUDE_DIRS}
  ${Python3_NumPy_INCLUDE_DIRS}
)

file (GLOB SRCS 
  "*.cpp"
  )

file (GLOB HEADERS 
  *.h *.hpp
  )

add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
add_compile_definitions(_WIN32_WINNT=0x0501)

add_library( EyeDetector SHARED ${SRCS} ${HEADERS} )
target_link_libraries( EyeDetector 
		       ${OpenCV_LIBS}
		       ${Boost_FILESYSTEM_LIBRARY}
		       ${Boost_SYSTEM_LIBRARY}
		       ${Boost_LIBRARIES}
           ${spii_LIBRARIES}
           "singleeyefitter"
           ${PYTHON_LIBRARIES}
           )
