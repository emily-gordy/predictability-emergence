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
- comment on or change file naming scheme to work for floats for lat/lon inputs
- add scoring to trainnn.py and remove from evalnn.py

testing/validation:

- how do we know if it is working?
- how do we know if it is broken?
- what is the minimum input needed to test each step versus full pipeline?

## Emily's notes on some things to look for in the outputs of trainnn and evalnn

`trainnn.py`

This script trains a batch of 10 CNNs at each grid point over land, outputting the trained model weights file (.pt) and some metrics on the validation data

inputs 
* lat, lon, seed
* assorted parser arguments
* landmask matrix (because of how we have updated this script, this is currently not being used. Originally it was a boolean mask indicating yes/no if gridpoint is over land. If not over land, the job is immediately terminated. Probably easier to incorporate this into the input csv of coordinates?)

outputs
* *.pt output is a pytorch model weights file
* *.json is a file of scores for the trained convolutional neural network (CNN) on the _validation_ data specified by lat, lon, seed
  * best_val_loss is the CNN's categorical cross-entropy loss on the validation data. It is a float that can have positive or negative values, but is usually positive. Note I don't use this as an input anywhere else but I would like to have it somewhere if needed.
  * val_accuracy is the CNN's accuracy on the testing data (number of correct predictions / total number of validation samples). It is a float on [0,1] and should be >0.5 but may not if only training for a few epochs
  * val_class_imbalance is (number of samples in the most common class / total number of validation samples). It is a float on [0.5,1]

`evalnn.py`

This script takes the metrics file generated from the training to select the best three performing CNNs at each gridpoint. It then pulls in those three CNNs, and inputs the testing data, saving some key metrics from that as well as a .pkl file of the raw predictions from each of the three CNNs.

inputs
* lat/lon coordinate
* assorted parser arguments (note some of the specific CNN hyperparameters can be removed here)

outputs
* .json file of metrics on the _testing_ data 
  * accuracy is the CNN's accuracy on the testing data, additionally using the discard test. It is a vector of 20 values on [0.5,1] and should ~generally~ increase i.e. last ~5 values should be greater than first ~5 values
  * test_imbalance is (number of samples in the most common class / total number of testing samples). It is a float on [0.5,1]
* .pkl of all predictions of on the testing data. If only doing one lat/lon at a time, it has dimension [3, ntestingsamples]. If grouping longitudes together into one latitude band, then it has dimension [nlongitude, 3, ntestingsamples]. The values in this array must be float [0,1].
* .pkl of the true outputtest values. It has dimension [ntestingsamples] (or [nlongitude, ntestingsamples] if grouping by latitude). The values in this array must be binary 0 or 1.

#### The Science!

I am currently working in the two jupyter notebooks `FirstPredict.ipynb` and `ObsPredict.ipynb` which have some descriptions about how I am using the outputs from the trained NNs and the testing predictions.

### questions for Emily

- what is the 'test' parameter in trainnn.py? it is used in line 262, it was originally `np.arange(38, 50)`
  - 'test' is a list of ensemble members to be used in the testing set. The data I am using is a 50 member ensemble of simulations so I split the data into training/validation by randomly grouping the first 38 members into 25 (n_train) members for training and 13 (n_val) members for validation (line 265). The final 12 members (i.e. members 38-50) are always used in testing. So, the test parameter does not change.
- in `DataHolder.py`, there are a couple of references to the `MPIInputOutput_SSPlist` object having an attribute `output_lat`/`output_lon` which breaks things
  - I don't know why this is breaking things, it should just be reading in a vector of lat/lon coordinates. I am going to dig into this
- related, I'm just somewhat confused what all the classes in DataHolder are doing with their functions
  - the DataHolder in general is a mess because I wrote the functions iteratively every time I updated the how I was constructing the input/output data. This is an artefact of me being not good at coding but I will go through and clean that up (in the spirit of cleaning everything else up)