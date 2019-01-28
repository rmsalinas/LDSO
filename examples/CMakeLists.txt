# TUM_MONO dataset
add_executable( run_dso_tum_mono run_dso_tum_mono.cc )
target_link_libraries( run_dso_tum_mono
  ldso ${THIRD_PARTY_LIBS} )

# EUROC dataset
add_executable( run_dso_euroc run_dso_euroc.cc )
target_link_libraries( run_dso_euroc
  ldso ${THIRD_PARTY_LIBS} )

# Kitti dataset
add_executable( run_dso_kitti run_dso_kitti.cc )
target_link_libraries( run_dso_kitti ldso ${THIRD_PARTY_LIBS} )


# Video input
add_executable( run_dso_video run_dso_video.cc video_reader.h)
target_link_libraries( run_dso_video ldso ${THIRD_PARTY_LIBS} )
