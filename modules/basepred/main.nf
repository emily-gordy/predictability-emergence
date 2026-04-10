process BASEPRED {    

    // Takes no input
    // input:

    // Output is a CSV file. emit gives the output a name, in this case a file
    output:
    path "*.csv", emit: baseline_prediction

    script:
    """
    basepred \
        --config ${params.common_config} \
        --data_dir ${params.data_dir} \
        --output_dir ./
    """
}
