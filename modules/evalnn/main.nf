process evalnn {

    input:
    // path 
    tuple val(lat), path(metricsFile), path(modelFile)

    output:
    path "*.npy", emit: model_pred
    // path "MPI_recordtemp_avgtime_${params.outputavgtime}_allssps_lat_${lat}_truetesting.pkl", emit: baseline_pred

    script:
    """
    evalnn --lat ${lat} \
        --config ${params.common_config} \
        --data_dir ${params.data_dir} \
        --model_dir . \
        --metrics_dir . \
        --output_dir ./
    """
}
