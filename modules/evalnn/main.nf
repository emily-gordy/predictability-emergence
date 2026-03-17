process evalnn {

    input:
    // path 

    output:
    path "MPI_recordtemp_avgtime_${params.outputavgtime}_allssps_lat_${lat}_testing.pkl", emit: model_pred
    path "MPI_recordtemp_avgtime_${params.outputavgtime}_allssps_lat_${lat}_truetesting.pkl", emit: baseline_pred

    script:
    """
    evalnn --lat ${lat} \
        --n_best ${params.n_best} \
        --outputavgtime ${params.outputavgtime} \
        --ssps ${params.ssps} \
        --experiment_era ${params.experiment_era} \
        --baseline_era ${params.baseline_era} \
        --input_length ${params.input_length} \
        --in_res ${params.in_res} \
        --out_res ${params.out_res} \
        --time_range ${params.time_range} \
        --file_front ${params.file_front} \
        --model_file_front ${params.model_file_front} \
        --input_var ${params.input_var} \
        --output_var ${params.output_var} \
        --n_train ${params.n_train} \
        --n_val ${params.n_val} \
        --test ${params.test} \
        --batch_size ${params.batch_size} \
        --lr ${params.lr} \
        --early_stopping_patience ${params.early_stopping_patience} \
        --epochs ${params.epochs} \
        --momentum ${params.momentum} \
        --data_dir ${params.data_dir}
    """
}
