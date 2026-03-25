process train_predict {
    conda "${moduleDir}/environment.yml"

    input:
    tuple val(id), val(lat), val(lon), val(seed)

    output:
    // tuple val(lat), val(lon), val(seed), stdout, emit: score
    tuple val(id), val(lat), val(lon), val(seed), path("MPI_recordtemp_avgtime_${params.outputavgtime}_allssps_lat_${lat}_lon_${lon}_seed_${seed}.json"), optional: true, emit: metrics
    tuple val(id), val(lat), val(lon), val(seed), path("MPI_recordtemp_avgtime_${params.outputavgtime}_allssps_lat_${lat}_lon_${lon}_seed_${seed}.pt"), optional: true, emit: model

    script:
    experiment_era = "${params.experiment_start} ${params.experiment_end}"
    baseline_era = "${params.baseline_start} ${params.baseline_end}"
    time_range = "${params.time_start} ${params.time_end}"
    test = "${params.test_start} ${params.test_end}"
    ssp_list = params.ssps.join(' ')

    """
    trainnn --lat ${lat} --seed ${seed} --lon ${lon} \
        --outputavgtime ${params.outputavgtime} \
        --ssps ${ssp_list} \
        --experiment_era ${experiment_era} \
        --baseline_era ${baseline_era} \
        --input_length ${params.input_length} \
        --in_res ${params.in_res} \
        --out_res ${params.out_res} \
        --time_range ${time_range} \
        --file_front ${params.file_front} \
        --model_file_front ${params.model_file_front} \
        --input_var ${params.input_var} \
        --output_var ${params.output_var} \
        --n_train ${params.n_train} \
        --n_val ${params.n_val} \
        --test ${test} \
        --batch_size ${params.batch_size} \
        --lr ${params.lr} \
        --early_stopping_patience ${params.early_stopping_patience} \
        --epochs ${params.epochs} \
        --momentum ${params.momentum} \
        --data_dir ${params.data_dir}
    """
}
