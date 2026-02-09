process processA {
    input:
    path input_file

    output:
    path 'model_output' into model
    path 'metrics_output' into metrics

    script:
    """
    # Placeholder for the actual processing commands
    echo "Processing ${input_file}"
    mkdir -p model_output metrics_output
    echo "Model results for ${input_file}" > model_output/results.txt
    echo "Metrics for ${input_file}" > metrics_output/metrics.txt
    """
}

workflow {

    main:
    ch_input = Channel.fromPath(params.input)

    processA(ch_input)



    publish:
        model = processA.out.model
        metrics = processA.out.metrics

}

output {
    model {
        path 'output/models'
    }

    metrics {
        path 'output/metrics'
    }
}