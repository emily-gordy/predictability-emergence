include { train_predict } from '../../modules/train_predict/main.nf'
include { basepred } from '../../modules/basepred/main.nf'
// include { evalnn } from '../../modules/evalnn/main.nf'

workflow MODEL_TRAINING {
    take:
    input_ch // This should be passed from the main workflow

    main:
    train_predict(input_ch)

    models = train_predict.out.model
            .map {
                id, lat, lon, seed, file ->
                [[id:id, lat:lat, lon:lon, seed:seed], file]
            }

    // metrics channel adjusted to group by (lat, lon) and pick top N within each group
    metrics = train_predict.out.metrics
        .map { id, lat, lon, seed, file ->
            // extract score from metrics JSON (same as your code)
            def score = file.splitJson()[4]['value']
            [[id:id, lat:lat, lon:lon, seed:seed], score as double, file]
        }
        // join to add the corresponding model path using the same key (the meta map)
        .join(models, by: [0], remainder: true)
        // Build tuples of: [ key=(lat,lon), payload={id,lat,lon,seed,score,metrics,model} ]
        .map { meta, score, metricsFile, modelFile ->
            def payload = [
                id     : meta.id,
                lat    : meta.lat,
                lon    : meta.lon,
                seed   : meta.seed,
                score  : score as double,
                metrics: metricsFile,
                model  : modelFile   // may be null if remainder=true and no match; handle as needed
            ]
            def key = [ meta.lat, meta.lon ]  // group key = (lat, lon)
            [ key, payload ]
        }
        // Group all records by (lat, lon)
        .groupTuple(by: 0)
        // For each group, sort by score desc and take top N; then emit those items
        .flatMap { key, items ->
            // items is a List<Map> of payloads for this (lat, lon)
            def top = items
                .sort { -it.score as double }   // descending
                .take(params.n_best as int)

            // Emit in your original tuple shape: [id, lat, lon, seed, score, metrics, model]
            top.collect { it -> [ it.id, it.lat, it.lon, it.seed, it.score, it.metrics, it.model ] }
        }
        .view { v -> "${v[3]} is a top model for ${v[1]}/${v[2]}: ${v[0]}"}

    // group over latitudes 

    // add evalnn step

    // calculate baselines 
    basepred()

    emit: // Specify the outputs you want published into the output directory
        model = train_predict.out.model
        metrics = train_predict.out.metrics

        baseline_pred = basepred.out.baseline_pred
        // model_pred = evalnn.out.model_pred

}