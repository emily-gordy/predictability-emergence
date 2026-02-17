include { MODEL_TRAINING } from './subworkflows/model_training/main.nf'

workflow {

    main:
    input_ch = channel.fromPath(params.input)
                .splitCsv( header: true )
                .map {
                    row -> [lon:row.LON, lat:row.LAT, seed:row.SEED]
                }
                // .view() // View the input channel to verify the data is being read correctly

    MODEL_TRAINING(input_ch)

    publish: // Specify the outputs you want published into the output directory
        model = MODEL_TRAINING.out.model
        metrics = MODEL_TRAINING.out.metrics
        // baseline_pred = MODEL_TRAINING.out.baseline_pred
        // model_pred = MODEL_TRAINING.out.model_pred
}

output {
    model { // Specify the output directory for the models
        path 'models'
    }

    metrics { // Specify the output directory for the metrics
        path 'metrics'
    }

    // baseline_pred { // Specify the output directory for the baseline predictions
    //     path 'predictions'
    // }

    // model_pred { // Specify the output directory for the model predictions
    //     path 'predictions'
    // }
}