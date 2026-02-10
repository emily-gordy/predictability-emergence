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
