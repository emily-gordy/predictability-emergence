## Nextflow pipeline

`nextflow run predict-emergence.nf -profile test,local`

### some settings for Mahuika

`module purge && module load Nextflow/25.10.2`

`export NXF_SYNTAX_PARSER='v2'` (use latest syntax for Nextflow)

### conda env notes

[You can specify existing conda environments](https://www.nextflow.io/docs/latest/conda.html#use-existing-conda-environments)

In `nextflow.config` there are a couple of relevant parameters.
`conda.enabled` just tells Nextflow to use conda environments.
`conda.cachedir` lets you set a cache dir (currently set to my nobackup dir, but change as you wish).

## Nextflow getting started resources

The Nextflow training materials just got updated.
[Intro course for building pipelines](https://training.nextflow.io/latest/hello_nextflow/) has fresh youtube videos to boot.

## Jen's notes on what's actually happening

(`/nesi/nobackup/uoa04506/predictability-emergence`)

`conda activate /nesi/nobackup/nesi99999/jreeve/predictability-emergence/ml-env`

Initial data prep done in `makedata.py`, sets up several pickled datasets (Gridded ERA5 Annual Mean and Summer and ERA5 Global Mean)

`trainnn.py` gets pickled data using `DataHolder.py`, does a bunch of prep, then weights/trains model and saves both a model file and metrics file.

`evalnn.py` takes all the metrics files for a given latitude (?) and finds the best models.
It then saves info about the best models in a new pickle dump.

### next steps

refactoring:

- add testing!
  - easy to set up test for eval step, just give some fake metrics
- currently doing a pickle dump from evalnn, what is the actual desired output?

testing/validation:

- how do we know if it is working?
- how do we know if it is broken?
- what is the minimum input needed to test each step versus full pipeline?
