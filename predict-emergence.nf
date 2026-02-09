include { train_predict } from './modules/train_predict/main.nf'

workflow {

    main:
    ch_input = channel.fromPath(params.input)
                .splitCsv( header: true )
                .map {
                    row -> [lon:row.LON, lat:row.LAT, seed:row.SEED]
                }

    train_predict(ch_input)

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