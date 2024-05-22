# Custom dispatch jobs for AIND ephys pipeline
## aind-ephys-job-dispatch


### Description

This simple capsule is designed to dispatch jobs for the AIND pipeline. 

It assumes the data is stored in the `data/` directory, and creates as many JSON files 
as the number of jobs that can be run in parallel. Each job consists of a recording with spiking activity that needs spike sorting.

### Inputs

The input data is assumed to be in the `data/` folder. The code shuold parse this folder and produce a list of 
`job_{i}.json` configuration files to be processed separately (e.g., different streams/groups, etc.)

### Parameters

The `code/run` script takes 2 arguments:

- `--concatenate`: `false` (default) | `true`. If `true`, the capsule will concatenate all recordings together. If `false`, each recording will be spike sorted separately.


### Output

The output of this capsule is a list of `job_{i}.json` JSON files in the `results/` folder, containing the parameters for a spike sorting job. 

Each JSON file contains the following fields:

- `session_name`: the session name (e.g., "ecephys_664438_2023-04-12_14-59-51")
- `recording_name`: the recording name, which will correspond to output folders downstreams (e.g, "experiment1_Record Node 101#Neuropix-PXI-100.probeA-AP_recording1")
- `recording_dict`: the SpikeInterface dict representation of the recording with paths relative to the `data` folder