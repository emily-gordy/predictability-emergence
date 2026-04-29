# Nextflow pipeline for predictability-emergence

## Quick start

On Mahuika:
```
conda activate /nesi/nobackup/nesi99999/jreeve/predictability-emergence/ml-env
module purge && module load Nextflow/25.10.2
export NXF_SYNTAX_PARSER='v2'
pip install -e .
nextflow run predict-emergence.nf -profile test,mahuika
```

## Installing the scripts

The nextflow pipeline requires scripts to be installed:

```
pip install -e .
```
Check that the commands 
```
trainnn -h
evalnn -h
basepred -h
```
work.

## Running tests outside of nextflow

You will need to have access to data.

You can run manual tests with
```
pytest tests --data_dir=<path/to/data>
```
or
```
DATA_DIR=<path/to/data> pytest tests
```

Notes: You can run a single test file, e.g. `pytest tests/test_trainnn.py`, or a single test inside a test file, e.g. `pytest tests/test_trainnn.py::test_small`. You will need about 10GB of memory to run this test. On Mahuika we recommend to run this test under SLURM.

## Commands to execute the pipeline

```
nextflow run predict-emergence.nf -profile test,local -with-dag test_workflow.png
```
This will launch a quick version of the workflow for testing locally. 

Should the workflow be interrupted for any reason, you can restart it with
```
nextflow run predict-emergence.nf -profile test,local -resume
```

To submit Mahuika using the SLURM scheduler
```
nextflow run predict-emergence.nf -profile test,mahuika -resume
```
To clean up the results
```
nextflow clean -free
```

## What is something goes wrong?

It is possible to inspect the output of some scripts to determine the cause of a potential failure. When executing the workflow you'll see 
something like `[c9/f4eb84]`. 
```
ls work/c9/f4eb84[TAB]
```
will show the produced files. In this directory, look for
```
ls work/c9/f4eb847fb16541735061ac92e99669/.command.*
```
to find stderr and other output messages.

## Adding a new task/process

Start by writing a module, e.g. under `modules/basepred/main.nf`. The process takes an input and produces an output
with a script. (In the case of `basepred`, there is no input and that section can be left out.) The ouput can be 
a path.

Decide whether the new process should be part of the subworkflow or the main workflow. If in the main workflow section
then: (1) include the module (top of the file) and add the call in the workflow section. Nextflow will figure out which
process can be run from the input/output. Note the `publish` field and `output` section, these determine what needs to 
be saved and where.

Example: adding evalnn

 1. Create `modules/evalnn/main.nf`
 2. Include `../../modules/evalnn/main.nf` in subworkflows/model_training/main.nf. Create an in[ut channel for `evalnn`. Add the the `evalnn` task. Emit the model prediction (`model_pred`).
 3. Save the `model_pred` in the predictions directory.
 

## Running on Mahuika

`module purge && module load Nextflow/25.10.2`

`export NXF_SYNTAX_PARSER='v2'` (use latest syntax for Nextflow)

## Nextflow getting started resources

The Nextflow training materials just got updated.
[Intro course for building pipelines](https://training.nextflow.io/latest/hello_nextflow/) has fresh youtube videos to boot.
