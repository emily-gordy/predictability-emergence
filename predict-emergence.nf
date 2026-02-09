include { train_predict } from './modules/train_predict/main.nf'

workflow {

    main:
    ch_input = Channel.fromPath(params.input)

    train_predict(ch_input)

    publish: // Specify the outputs you want published into the output directory
        model = train_predict.out.model
        metrics = train_predict.out.metrics

}

output {
    model {
        path 'output/models' 
    }

    metrics {
        path 'output/metrics'
    }
}