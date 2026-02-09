process train_predict {
    conda "${moduleDir}/environment.yml"

    input:
    tuple val(lat), val(lon), val(seed)

    output:
    path('MPI_recordtemp_avgtime_${params.outputavgtime}_allssps_lat_${lat}_lon_${lon}_seed_${seed}.json'), emit: metrics
    path('MPI_recordtemp_avgtime_${params.outputavgtime}_allssps_lat_${lat}_lon_${lon}_seed_${seed}.pt'), emit: model

    script:
    """
    trainnn.py --lat ${lat} --seed ${seed} --lon ${lon} --outputavgtime ${params.outputavgtime}
    """
}
