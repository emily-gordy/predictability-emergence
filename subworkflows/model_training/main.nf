include { train_predict } from '../../modules/train_predict/main.nf'
include { evalnn } from '../../modules/evalnn/main.nf'

workflow MODEL_TRAINING {
    take:
    input_ch // This should be passed from the main workflow

    main:
    train_predict(input_ch)

    models = train_predict.out.model
            .map {
                lat, lon, seed, file ->
                [[lat:lat, lon:lon, seed:seed], file]
            }
    metrics = train_predict.out.metrics
                .map {
                    lat, lon, seed, file ->
                    [[lat:lat, lon:lon, seed:seed], file.splitJson()[4]['value'], file] // pull the score out of the metrics file
                }
                .join(models, by: [0], remainder: true) // add the path for the model file
                .toSortedList { a, b -> b[1] <=> a[1] } // sort by the score
                .flatMap()
                .take( params.n_best ) // get the best #
                // .view()
    // add evalnn step
    evalnn(metrics)

    emit: // Specify the outputs you want published into the output directory
        model = train_predict.out.model
        metrics = train_predict.out.metrics
        // baseline_pred = evalnn.out.baseline_pred
        // model_pred = evalnn.out.model_pred

}