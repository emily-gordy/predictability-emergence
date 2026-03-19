process train_predict {

    input:
    tuple val(id), val(lat), val(lon), val(seed)

    output:
    // tuple val(lat), val(lon), val(seed), stdout, emit: score
    tuple val(id), val(lat), val(lon), val(seed), path("MPI_recordtemp_avgtime_${params.outputavgtime}_allssps_lat_${lat}_lon_${lon}_seed_${seed}.json"), emit: metrics
    tuple val(id), val(lat), val(lon), val(seed), path("MPI_recordtemp_avgtime_${params.outputavgtime}_allssps_lat_${lat}_lon_${lon}_seed_${seed}.pt"), emit: model

    script:
    experiment_era = "${params.experiment_start} ${params.experiment_end}"
    baseline_era = "${params.baseline_start} ${params.baseline_end}"
    time_range = "${params.time_start} ${params.time_end}"
    test = "${params.test_start} ${params.test_end}"
    ssp_list = params.ssps.join(' ')

    """
    trainnn --lat ${lat} --seed ${seed} --lon ${lon} \
        --config ${params.common_config} \
        --n_train ${params.n_train} \
        --n_val ${params.n_val} \
        --batch_size ${params.batch_size} \
        --lr ${params.lr} \
        --early_stopping_patience ${params.early_stopping_patience} \
        --epochs ${params.epochs} \
        --momentum ${params.momentum} \
        --data_dir ${params.data_dir}
    """
}
