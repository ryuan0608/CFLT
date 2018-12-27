SET(HEADERS
  src/include/feature_descriptor.h
  src/include/ldmarkmodel.h
  src/include/helper.h
  src/include/hog.h
)

SET(SOURCE
  src/feature_descriptor.cpp
  src/helper.cpp
  src/hog.c
  src/ldmarkmodel.cpp
)

include_directories("src/include/cereal/")
include_directories("src/include/cereal/archives")
include_directories("src/include/cereal/details")
include_directories("src/include/cereal/external")
include_directories("src/include/cereal/external/rapidjson")
include_directories("src/include/cereal/external/rapidxml/")
include_directories("src/include/cereal/types")

add_library(HyperLandmarks ${HEADERS} ${SOURCE})

target_link_libraries(HyperLandmarks ${OpenCV_LIBS})

