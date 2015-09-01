cmake_minimum_required(VERSION 2.8)
project( FT )
find_package( OpenCV REQUIRED )

add_executable( smoothFaceTracking smooth_face_tracking.cpp )
add_executable( basicFaceTracking basic_face_detection.cpp )
add_executable( improvedFaceTracking improved_face_detection.cpp )
target_link_libraries( smoothFaceTracking ${OpenCV_LIBS} )
target_link_libraries( basicFaceTracking ${OpenCV_LIBS} )
target_link_libraries( improvedFaceTracking ${OpenCV_LIBS} )


