process basepred {

    input:
    // path 

    output:
    path "prediction/*.csv", emit: baseline_pred // possibly refine later

    script:
    """
    basepred \
        --config ${params.common_config} \
        --data_dir ${params.data_dir}
    """
}
