# Generate the example feature and target data files for testing
foreach(TIMESTEP IN ITEMS 1 2 3)
    # Generate files for 1D and 2D feature data
    foreach(DIM IN ITEMS 1d 2d)
        set(OUTPUT_FILE test_feature${DIM}_${TIMESTEP}.nc)
        set(INPUT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/test_feature${DIM}_${TIMESTEP}.cdl)
        add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND ncgen -k netCDF-4 -o ${OUTPUT_FILE} ${INPUT_FILE}
            DEPENDS ${INPUT_FILE}
        )
        list(APPEND GENERATED_FILES ${OUTPUT_FILE})
    endforeach()

    # Generate files for target data
    set(OUTPUT_FILE ncsteps_${TIMESTEP}.nc)
    set(INPUT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ncsteps_${TIMESTEP}.cdl)
    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND ncgen -k netCDF-4 -o ${OUTPUT_FILE} ${INPUT_FILE}
        DEPENDS ${INPUT_FILE}
    )
    list(APPEND GENERATED_FILES ${OUTPUT_FILE})
endforeach()
add_custom_target(generate_input_files ALL DEPENDS ${GENERATED_FILES})
