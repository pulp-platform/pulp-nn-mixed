project(PULPNNMIXED)

add_library(pulp-nn-mixed STATIC)

target_compile_options(pulp-nn-mixed PRIVATE
  -Ofast
  -Wno-conversion
  -Wno-sign-compare
  -Wno-sign-conversion
  -Wno-unused-variable
  -Wno-unused-function
  -Wno-unused-parameter
  -Wno-incompatible-pointer-types
  -Wno-implicit-function-declaration
  -Wno-attributes
  -Wno-discarded-qualifiers
  -Wno-pointer-sign
  -Wno-unused-value
  -Wno-int-conversion
  -Wno-typedef-redefinition
  -Wno-uninitialized
)

target_include_directories(pulp-nn-mixed PRIVATE ${PULP_SDK_INCLUDES})
target_compile_options(pulp-nn-mixed PRIVATE ${PULP_SDK_COMPILE_FLAGS})

if(PULPNNVERSION STREQUAL XPULPV2)
  if(PULPNNBITWIDTH STREQUAL 32)
    set(INCLUDES
      ${CMAKE_CURRENT_LIST_DIR}/XpulpV2/32bit/include
    )
    file(GLOB_RECURSE SRC
      XpulpV2/32bit/src/**/*.c
    )
  endif()
endif()

target_include_directories(pulp-nn-mixed PUBLIC
  ${INCLUDES}
)
target_sources(pulp-nn-mixed PRIVATE ${SRC})
set(PULPNN_INCLUDES ${INCLUDES} CACHE INTERNAL "PULPNN_INCLUDES")
