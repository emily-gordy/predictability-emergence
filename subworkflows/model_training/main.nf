include { train_predict } from '../../modules/train_predict/main.nf'
include { evalnn } from '../../modules/evalnn/main.nf'

workflow MODEL_TRAINING {
    take:
    input_ch // This should be passed from the main workflow

    main:
    train_predict(input_ch)

    scores = train_predict.out.metrics
                .map {
                    lat, lon, seed, file ->
                    [[lat, lon, seed], file.splitJson()[4]]
                }
                .map {
                    meta, json ->
                    [meta + json]
                }
                // .splitJson()
                // .map { v -> v.score }
                .view()

    // add evalnn step

    emit: // Specify the outputs you want published into the output directory
        model = train_predict.out.model
        metrics = train_predict.out.metrics
        // baseline_pred = evalnn.out.baseline_pred
        // model_pred = evalnn.out.model_pred

}