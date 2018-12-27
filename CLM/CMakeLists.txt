SET(HEADERS
  include/CCNF_patch_expert.hpp
  include/CLM_utils.hpp
  include/CLM.hpp
  include/CLMParameters.hpp
  include/LandmarkMapper.hpp
  include/CLMConfig.h
  include/PatchExperts.hpp
  include/PDM.hpp
  include/SVR_patch_expert.hpp
  include/LandmarkLocaliser.hpp
)

SET(SOURCE
  src/CCNF_patch_expert.cpp
  src/CLM_utils.cpp
  src/CLM.cpp
  src/CLMParameters.cpp
  src/LandmarkMapper.cpp
  src/PatchExperts.cpp
  src/PDM.cpp
  src/SVR_patch_expert.cpp
  src/LandmarkLocaliser.cpp
)

include_directories("include/")
include_directories("src/")

add_library(CLM ${HEADERS} ${SOURCE})

target_link_libraries(CLM ${OpenCV_LIBS})

