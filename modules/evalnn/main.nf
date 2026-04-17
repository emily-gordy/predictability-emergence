process evalnn {

    input:
    // need the model and the metrics
    tuple val(lat), path(metricsFile), path(modelFile)

    output:
    path "*.npy", emit: model_pred

    script:
    """
    evalnn --lat ${lat} \
        --n_best ${params.n_best} \
        --config ${params.common_config} \
        --data_dir ${params.data_dir} \
        --model_dir . \
        --metrics_dir . \
        --output_dir ./
    """
}
