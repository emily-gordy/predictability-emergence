include { train_predict } from './modules/train_predict/main.nf'

workflow {

    main:
    input_ch = channel.fromPath(params.input)
                .splitCsv( header: true )
                .map {
                    row -> [lon:row.LON, lat:row.LAT, seed:row.SEED]
                }
                // .view() // View the input channel to verify the data is being read correctly

    train_predict(input_ch)

    

    publish: // Specify the outputs you want published into the output directory
        model = train_predict.out.model
        metrics = train_predict.out.metrics

}

output {
    model { // Specify the output directory for the models
        path 'models'
    }

    metrics { // Specify the output directory for the metrics
        path 'metrics'
    }
}