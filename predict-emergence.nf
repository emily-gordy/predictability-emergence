include { MODEL_TRAINING } from './subworkflows/model_training/main.nf'

/**
 * Strict integer parser for string fields (with trim).
 * Throws IllegalArgumentException with a clear field name.
 */
int parseIntStrict(String s, String name) {
    if (s == null) {
        throw new IllegalArgumentException("${name} is null")
    }
    String t = s.trim()
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
 * Output format (padded longitude): <abs(lat)><N|S><lon_3digits><seed>
 * Examples:
 *   makeId("90",  "90",  "100")  -> "90N090100"
 *   makeId("-90", "180", "100")  -> "90S180100"
 *   makeId("0",   "0",   "1")    -> "0N0001"
 */
String makeId(String latStr, String lonStr, String seedStr) {
    Integer lat  = parseIntStrict(latStr,  'lat')
    Integer lon  = parseIntStrict(lonStr,  'lon')
    Integer seed = parseIntStrict(seedStr, 'seed')

    if (lat < -90 || lat > 90) {
        throw new IllegalArgumentException("Latitude must be in [-90, 90], got: ${lat}")
    }
    if (lon < 0 || lon > 360) {
        throw new IllegalArgumentException("Longitude must be in [0, 360], got: ${lon}")
    }

    String hemi    = (lat >= 0) ? 'N' : 'S'
    Integer absLat     = Math.abs(lat)
    String lonPart = String.format("%03d", lon)   // <-- 3-digit padded longitude

    return "${absLat}${hemi}${lonPart}${seed}"
}

/**
 * Parse an ID of the padded form: <abs(lat)><N|S><lon_3digits><seed>
 *   - abs(lat): 1–2 digits in [0, 90]
 *   - hemisphere: 'N' or 'S'
 *   - lon: exactly 3 digits in [000, 360]
 *   - seed: remaining digits (>=1 digit)
 *
 * By default this enforces the padded format. If you still have legacy
 * unpadded IDs, pass allowLegacyUnpadded = true to also accept them.
 *
 * Returns: [lat: int, lon: int, seed: int]
 */
Map parseId(String id, boolean allowLegacyUnpadded = false) {
    if (id == null) throw new IllegalArgumentException("id is null")
    String s = id.trim()
    if (!s) throw new IllegalArgumentException("id is empty")

    // First try strict padded pattern: lon is exactly 3 digits
    def m = (s =~ /^(\d{1,2})([NS])(\d{3})(\d+)$/)
    if (!m.matches() && allowLegacyUnpadded) {
        // Legacy (unpadded lon: 1–3 digits) — fallback matcher
        m = (s =~ /^(\d{1,2})([NS])(0|[1-9]\d{0,2})(\d+)$/)
    }

    if (!m.matches()) {
        throw new IllegalArgumentException(
            "ID does not match expected pattern <abs(lat)><N|S><lon(3d)><seed>: '${id}'"
        )
    }

    Integer absLat = Integer.parseInt(m.group(1))
    String hemi = m.group(2)
    Integer lon     = Integer.parseInt(m.group(3))
    String seedStr = m.group(4)

    if (absLat < 0 || absLat > 90) {
        throw new IllegalArgumentException("Latitude magnitude must be in [0, 90], got: ${absLat}")
    }
    if (lon < 0 || lon > 360) {
        throw new IllegalArgumentException("Longitude must be in [0, 360], got: ${lon}")
    }

    Integer lat = (hemi == 'N') ? absLat : -absLat
    Integer seed
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
                    row -> [id:makeId(row.LAT,row.LON,row.SEED), lat:row.LAT, lon:row.LON, seed:row.SEED]
                }
                // .view() // View the input channel to verify the data is being read correctly

    MODEL_TRAINING(input_ch)

    publish: // Specify the outputs you want published into the output directory
        model = MODEL_TRAINING.out.model
        metrics = MODEL_TRAINING.out.metrics
        baseline_pred = MODEL_TRAINING.out.baseline_pred
        model_pred = MODEL_TRAINING.out.model_pred // output of evalnn
}

output {
    model { // Specify the output directory for the models
        path 'models'
    }

    metrics { // Specify the output directory for the metrics
        path 'metrics'
    }

    baseline_pred { // Specify the output directory for the baseline predictions
        path 'predictions'
    }

    model_pred { // Specify the output directory for the model predictions
         path 'predictions'
    }
}
