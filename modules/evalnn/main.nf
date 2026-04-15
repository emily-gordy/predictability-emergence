process evalnn {

    input:
    // path 

    output:
    path "MPI_recordtemp_avgtime_${params.outputavgtime}_allssps_lat_${lat}_testing.pkl", emit: model_pred
    // path "MPI_recordtemp_avgtime_${params.outputavgtime}_allssps_lat_${lat}_truetesting.pkl", emit: baseline_pred

    script:
    """
    evalnn --lat ${lat} \
        --n_best ${params.n_best} \
        --config ${params.common_config}
        --test ${params.test} \
        --batch_size ${params.batch_size} \
        --lr ${params.lr} \
        --early_stopping_patience ${params.early_stopping_patience} \
        --epochs ${params.epochs} \
        --momentum ${params.momentum} \
        --data_dir ${params.data_dir}
    """
}
