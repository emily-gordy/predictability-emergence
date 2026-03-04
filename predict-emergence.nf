include { MODEL_TRAINING } from './subworkflows/model_training/main.nf'

/**
 * Strict integer parser for string fields (with trim).
 * Throws IllegalArgumentException with a clear field name.
 */
int parseIntStrict(String s, String name) {
    if (s == null) {
        throw new IllegalArgumentException("${name} is null")
    }
    final t = s.trim()
    if (!t) {
        throw new IllegalArgumentException("${name} is empty")
    }
    if (!(t ==~ /^[-+]?\d+$/)) {
        throw new IllegalArgumentException("${name} must be an integer string, got: '${s}'")
    }
    try {
        return Integer.parseInt(t)
    } catch (NumberFormatException e) {
        throw new IllegalArgumentException("${name} is out of 32-bit int range: '${s}'")
    }
}

/**
 * Build an ID string from string inputs with domain checks:
 *   lat ∈ [-90, 90], lon ∈ [0, 360], seed integer.
 * Output format: <abs(lat)><N|S><lon><seed>
 * Examples:
 *   makeId("90", "90", "100")   -> "90N90100"
 *   makeId("-90", "180", "100") -> "90S180100"
 */
String makeId(String latStr, String lonStr, String seedStr) {
    final int lat  = parseIntStrict(latStr,  'lat')
    final int lon  = parseIntStrict(lonStr,  'lon')
    final int seed = parseIntStrict(seedStr, 'seed')

    if (lat < -90 || lat > 90) {
        throw new IllegalArgumentException("Latitude must be in [-90, 90], got: ${lat}")
    }
    if (lon < 0 || lon > 360) {
        throw new IllegalArgumentException("Longitude must be in [0, 360], got: ${lon}")
    }

    final String hemi = (lat >= 0) ? 'N' : 'S'
    final int absLat  = Math.abs(lat)

    return "${absLat}${hemi}${lon}${seed}"
}

/**
 * Parse an ID of the form: <abs(lat)><N|S><lon><seed>
 *   - abs(lat): 1–2 digits in [0,90]
 *   - hemisphere: 'N' or 'S'
 *   - lon: 1–3 digits in [0,360]
 *   - seed: remaining digits (>=1 digit)
 *
 * Examples:
 *   "90N90100"  -> [lat: 90, lon: 90,  seed: 100]
 *   "90S180100" -> [lat:-90, lon: 180, seed: 100]
 *
 * @throws IllegalArgumentException if the string is malformed or out of range
 */
Map parseId(String id) {
    if (id == null) {
        throw new IllegalArgumentException("id is null")
    }
    final s = id.trim()
    if (!s) {
        throw new IllegalArgumentException("id is empty")
    }

    // Regex groups:
    //  1: abs lat (1–2 digits)
    //  2: hemisphere (N|S)
    //  3: lon (0 OR non-zero number up to 3 digits)
    //  4: seed (1+ digits)
    def m = (s =~ /^(\d{1,2})([NS])(0|[1-9]\d{0,2})(\d+)$/)
    if (!m.matches()) {
        throw new IllegalArgumentException(
            "ID does not match expected pattern <abs(lat)><N|S><lon><seed>: '${id}'"
        )
    }

    final int absLat = Integer.parseInt(m.group(1))
    final String hemi = m.group(2)
    final int lon     = Integer.parseInt(m.group(3))
    final String seedStr = m.group(4)

    // Range checks
    if (absLat < 0 || absLat > 90) {
        throw new IllegalArgumentException("Latitude magnitude must be in [0, 90], got: ${absLat}")
    }
    if (lon < 0 || lon > 360) {
        throw new IllegalArgumentException("Longitude must be in [0, 360], got: ${lon}")
    }

    // Reconstruct signed latitude
    final int lat = (hemi == 'N') ? absLat : -absLat

    // Seed: parse as int — if you expect very large seeds, keep it as String
    final int seed
    try {
        seed = Integer.parseInt(seedStr)
    } catch (NumberFormatException e) {
        throw new IllegalArgumentException("Seed is out of 32-bit int range: '${seedStr}'")
    }

    return [ lat: lat, lon: lon, seed: seed ]
}

workflow {

    main:
    input_ch = channel.fromPath(params.input)
                .splitCsv( header: true )
                .map {
                    row -> [id:makeId(row.LAT,row.LON,row.SEED), lon:row.LON, lat:row.LAT, seed:row.SEED]
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