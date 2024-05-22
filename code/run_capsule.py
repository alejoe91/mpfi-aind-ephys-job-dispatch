import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# GENERAL IMPORTS
import argparse
import numpy as np
from pathlib import Path
import json


# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.extractors as se
import probeinterface as pi

from spikeinterface.core.core_tools import SIJsonEncoder


data_folder = Path("../data")
results_folder = Path("../results")
results_folder.mkdir(exist_ok=True, parents=True)

# Define argument parser
parser = argparse.ArgumentParser(description="Dispatch jobs for AIND ephys pipeline")

concat_group = parser.add_mutually_exclusive_group()
concat_help = "Whether to concatenate recordings (segments) or not. Default: False"
concat_group.add_argument("--concatenate", action="store_true", help=concat_help)
concat_group.add_argument("static_concatenate", nargs="?", default="false", help=concat_help)


if __name__ == "__main__":
    args = parser.parse_args()

    CONCAT = True if args.static_concatenate and args.static_concatenate.lower() == "true" else args.concatenate

    print(f"Running job dispatcher with the following parameters:")
    print(f"\tCONCATENATE RECORDINGS: {CONCAT}")

    # THIS IS THE MAIN PART OF THE SCRIPT
    #
    # This script assumes the data are in the ../data/ecephys folder and needs to:
    # 1. Read the different streams/recordings that require parallel processing
    # 2. Create a job config file for each stream/segment, which includes:
    #    - session_name
    #    - recording_name
    #    - recording_dict: the recording object serialized to a dictionary by the to_dict method
    # 3. Save the job config files in the ../results folder as job_{i}.json

    job_dict_list = []

    # EXAMPLE: MPFI data
    #
    # SpikeGLX folder with:
    # - Neuropixels streams: ap streams need to be processed in parallel
    # - NIDQ streams: containing data from a silicon probe
    # - probe_nidq.json: probeinterface probe file with the silicon probe layout

    # get blocks/experiments and streams info
    spikeglx_folders = [p for p in data_folder.iterdir() if p.is_dir()]
    print(spikeglx_folders)
    assert len(spikeglx_folders) == 1, "Attach one SpikeGLX folder at a time"
    spikeglx_folder = spikeglx_folders[0]
    session_name = spikeglx_folder.name
    stream_names, stream_ids = se.get_neo_streams("spikeglx", spikeglx_folder)

    # spikeglx has only one block
    num_blocks = 1
    block_index = 0

    print(f"\tNum. Blocks {num_blocks} - Num. streams: {len(stream_names)}")
    print("\tRecording to be processed in parallel:")
    for stream_name in stream_names:
        if "lf" in stream_name:
            continue
        recording = se.read_spikeglx(spikeglx_folder, stream_name=stream_name)

        if "nidq" in stream_name:
            probe_json_file = spikeglx_folder / "probe_nidq.json"
            probegroup = pi.read_probeinterface(probe_json_file)
            print(f"\t\tSetting probe file for {stream_name}")
            recording = recording.set_probegroup(probegroup, group_mode="by_shank")

        if CONCAT:
            recordings = [recording]
        else:
            recordings = si.split_recording(recording)

        HAS_CHANNEL_GROUPS = len(np.unique(recording.get_channel_groups())) > 1

        for i_r, recording in enumerate(recordings):
            if CONCAT:
                recording_name = f"block{block_index}_{stream_name}_recording"
            else:
                recording_name = f"block{block_index}_{stream_name}_recording{i_r + 1}"

            total_duration = np.round(recording.get_total_duration(), 2)

            if HAS_CHANNEL_GROUPS:
                for group_name, recording_group in recording.split_by("group").items():
                    recording_name_group = f"{recording_name}_group{group_name}"
                    print(
                        f"\t\t{recording_name_group} - Duration: {total_duration} s - Num. channels: {recording_group.get_num_channels()}"
                    )
                    job_dict = dict(
                        session_name=session_name,
                        recording_name=str(recording_name_group),
                        recording_dict=recording_group.to_dict(recursive=True, relative_to=data_folder),
                    )
                    job_dict_list.append(job_dict)
            else:
                print(
                    f"\t\t{recording_name} - Duration: {total_duration} s - Num. channels: {recording.get_num_channels()}"
                )

                job_dict = dict(
                    session_name=session_name,
                    recording_name=str(recording_name),
                    recording_dict=recording.to_dict(recursive=True, relative_to=data_folder),
                )
                job_dict_list.append(job_dict)

    for i, job_dict in enumerate(job_dict_list):
        with open(results_folder / f"job_{i}.json", "w") as f:
            json.dump(job_dict, f, indent=4, cls=SIJsonEncoder)
    print(f"Generated {len(job_dict_list)} job config files")
