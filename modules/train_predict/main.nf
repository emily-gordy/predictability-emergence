process train_predict {
    conda "${moduleDir}/environment.yml"

    input:
    tuple val(lat), val(lon), val(seed)

    output:
    path('*.json')

    script:
    """
    trainnn.py --lat ${lat} --seed ${seed} --lon ${lon}
    """
}
