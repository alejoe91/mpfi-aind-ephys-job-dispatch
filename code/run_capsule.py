import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# GENERAL IMPORTS
import sys
import argparse
import numpy as np
from pathlib import Path
import json
import logging


# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.extractors as se
import probeinterface as pi

from spikeinterface.core.core_tools import SIJsonEncoder

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

data_folder = Path("../data")
results_folder = Path("../results")
results_folder.mkdir(exist_ok=True, parents=True)

MAX_NUM_NEGATIVE_TIMESTAMPS = 10
MAX_TIMESTAMPS_DEVIATION_MS = 1

# Define argument parser
parser = argparse.ArgumentParser(description="Dispatch jobs for AIND ephys pipeline")

concat_group = parser.add_mutually_exclusive_group()
concat_help = "Whether to concatenate recordings (segments) or not. Default: False"
concat_group.add_argument("--concatenate", action="store_true", help=concat_help)
concat_group.add_argument("static_concatenate", nargs="?", default="false", help=concat_help)

split_group = parser.add_mutually_exclusive_group()
split_help = "Whether to process different groups separately"
split_group.add_argument("--split-groups", action="store_true", help=split_help)
split_group.add_argument("static_split_groups", nargs="?", default="false", help=split_help)

debug_group = parser.add_mutually_exclusive_group()
debug_help = "Whether to run in DEBUG mode"
debug_group.add_argument("--debug", action="store_true", help=debug_help)
debug_group.add_argument("static_debug", nargs="?", default="false", help=debug_help)

debug_duration_group = parser.add_mutually_exclusive_group()
debug_duration_help = (
    "Duration of clipped recording in debug mode. Default is 30 seconds. Only used if debug is enabled"
)
debug_duration_group.add_argument("--debug-duration", default=30, help=debug_duration_help)
debug_duration_group.add_argument("static_debug_duration", nargs="?", default=None, help=debug_duration_help)


if __name__ == "__main__":
    args = parser.parse_args()

    CONCAT = True if args.static_concatenate and args.static_concatenate.lower() == "true" else args.concatenate
    SPLIT_GROUPS = (
        True if args.static_split_groups and args.static_split_groups.lower() == "true" else args.split_groups
    )
    DEBUG = args.debug or args.static_debug.lower() == "true"
    DEBUG_DURATION = float(args.static_debug_duration or args.debug_duration)

    logging.info(f"Running job dispatcher with the following parameters:")
    logging.info(f"\tCONCATENATE RECORDINGS: {CONCAT}")
    logging.info(f"\tSPLIT GROUPS: {SPLIT_GROUPS}")
    logging.info(f"\tDEBUG: {DEBUG}")
    logging.info(f"\tDEBUG DURATION: {DEBUG_DURATION}")

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
    logging.info(spikeglx_folders)
    assert len(spikeglx_folders) == 1, "Attach one SpikeGLX folder at a time"
    spikeglx_folder = spikeglx_folders[0]
    session_name = spikeglx_folder.name
    stream_names, stream_ids = se.get_neo_streams("spikeglx", spikeglx_folder)

    # spikeglx has only one block
    num_blocks = 1
    block_index = 0

    logging.info(f"\tNum. Blocks {num_blocks} - Num. streams: {len(stream_names)}")
    recording_dict = {}
    for stream_name in stream_names:
        if "lf" in stream_name:
            continue
        recording_name = f"block{block_index}_{stream_name}_recording"
        recording = se.read_spikeglx(spikeglx_folder, stream_name=stream_name)

        if "nidq" in stream_name:
            probe_json_file = spikeglx_folder / "probe_nidq.json"
            probegroup = pi.read_probeinterface(probe_json_file)
            logging.info(f"\t\tSetting probe file for {stream_name}")
            recording = recording.set_probegroup(probegroup, group_mode="by_shank")

        recording_dict[(session_name, recording_name)] = {}
        recording_dict[(session_name, recording_name)]["raw"] = recording

        # load the associated LF stream (if available)
        if "ap" in stream_name:
            stream_name_lf = stream_name.replace("ap", "lf")
            try:
                recording_lf = se.read_spikeglx(spikeglx_folder, stream_name=stream_name_lf)
                recording_dict[(session_name, recording_name)]["lfp"] = recording_lf
            except:
                logging.info(f"\t\tNo LFP stream found for {stream_name}")

    # populate job dict list
    job_dict_list = []
    logging.info("Recording to be processed in parallel:")
    for session_recording_name in recording_dict:
        session_name, recording_name = session_recording_name
        recording = recording_dict[session_recording_name]["raw"]
        recording_lfp = recording_dict[session_recording_name].get("lfp", None)

        HAS_LFP = recording_lfp is not None
        if CONCAT:
            recordings = [recording]
            recordings_lfp = [recording_lfp] if HAS_LFP else None
        else:
            recordings = si.split_recording(recording)
            recordings_lfp = si.split_recording(recording_lfp) if HAS_LFP else None

        for recording_index, recording in enumerate(recordings):
            if not CONCAT:
                recording_name_segment = f"{recording_name}{recording_index + 1}"
            else:
                recording_name_segment = f"{recording_name}"

            if HAS_LFP:
                recording_lfp = recordings_lfp[recording_index]

            # timestamps should be monotonically increasing, but we allow for small glitches
            skip_times = False
            for segment_index in range(recording.get_num_segments()):
                times = recording.get_times(segment_index=segment_index)
                times_diff = np.diff(times)
                num_negative_times = np.sum(times_diff < 0)

                if num_negative_times > 0:
                    logging.info(f"\t\t{recording_name} - Times not monotonically increasing.")
                    if num_negative_times > MAX_NUM_NEGATIVE_TIMESTAMPS:
                        logging.info(
                            f"\t\t{recording_name} - Skipping timestamps for too many negative timestamps: {num_negative_times}"
                        )
                        skip_times = True
                        break
                    if np.max(np.abs(times_diff)) * 1000 > MAX_TIMESTAMPS_DEVIATION_MS:
                        logging.info(
                            f"\t\t{recording_name} - Skipping timesstamps for too large deviation: {np.max(np.abs(times_diff))} ms"
                        )
                        skip_times = True
                        break

            if skip_times:
                recording.reset_times()

            if DEBUG:
                recording_list = []
                for segment_index in range(recording.get_num_segments()):
                    recording_one = si.split_recording(recording)[segment_index]
                    recording_one = recording_one.frame_slice(
                        start_frame=0,
                        end_frame=min(int(DEBUG_DURATION * recording.sampling_frequency), recording_one.get_num_samples())
                    )
                    recording_list.append(recording_one)
                recording = si.append_recordings(recording_list)
                if HAS_LFP:
                    recording_lfp_list = []
                    for segment_index in range(recording_lfp.get_num_segments()):
                        recording_lfp_one = si.split_recording(recording_lfp)[segment_index]
                        recording_lfp_one = recording_lfp_one.frame_slice(
                            start_frame=0,
                            end_frame=min(int(DEBUG_DURATION * recording_lfp.sampling_frequency), recording_lfp_one.get_num_samples())
                        )
                        recording_lfp_list.append(recording_lfp_one)
                    recording_lfp = si.append_recordings(recording_lfp_list)

            duration = np.round(recording.get_total_duration(), 2)

            # if multiple channel groups, process in parallel
            if SPLIT_GROUPS and len(np.unique(recording.get_channel_groups())) > 1:
                for group_name, recording_group in recording.split_by("group").items():
                    recording_name_group = f"{recording_name_segment}_group{group_name}"
                    job_dict = dict(
                        session_name=session_name,
                        recording_name=str(recording_name_group),
                        recording_dict=recording_group.to_dict(recursive=True, relative_to=data_folder),
                        skip_times=skip_times,
                        duration=duration,
                        debug=DEBUG,
                    )
                    rec_str = f"\t{recording_name_group} - Duration: {duration} s - Num. channels: {recording_group.get_num_channels()}"
                    if HAS_LFP:
                        recording_lfp_group = recording_lfp.split_by("group")[group_name]
                        job_dict["recording_lfp_dict"] = recording_lfp_group.to_dict(
                            recursive=True, relative_to=data_folder
                        )
                        rec_str += f" (with LFP stream)"
                    logging.info(rec_str)
                    job_dict_list.append(job_dict)
            else:
                job_dict = dict(
                    session_name=session_name,
                    recording_name=str(recording_name_segment),
                    recording_dict=recording.to_dict(recursive=True, relative_to=data_folder),
                    skip_times=skip_times,
                    duration=duration,
                    debug=DEBUG,
                )
                rec_str = f"\t{recording_name_segment} - Duration: {duration} s - Num. channels: {recording.get_num_channels()}"
                if HAS_LFP:
                    job_dict["recording_lfp_dict"] = recording_lfp.to_dict(recursive=True, relative_to=data_folder)
                    rec_str += f" (with LFP stream)"
                logging.info(rec_str)
                job_dict_list.append(job_dict)

    if not results_folder.is_dir():
        results_folder.mkdir(parents=True)

    for i, job_dict in enumerate(job_dict_list):
        with open(results_folder / f"job_{i}.json", "w") as f:
            json.dump(job_dict, f, indent=4, cls=SIJsonEncoder)
    logging.info(f"Generated {len(job_dict_list)} job config files")
