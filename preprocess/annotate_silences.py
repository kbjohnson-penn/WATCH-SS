import argparse
import json
import pandas as pd

def annotate_silences_whisper(transcript_file, audio_file):
    """
    Annotate Whisper transcript with silences in the audio.

    args:
        transcript_file (str): Path to the transcript file.
        audio_file (str): Path to the audio file.
    """
    if not transcript_file.endswith(".json"):
        raise ValueError("This method only supports JSON Whisper transcripts.")

    with open(transcript_file, "r") as file:
        transcript = json.load(file)

    return

def annotate_silences_datagain(transcript_file, audio_file):
    """
    Annotate Datagain transcript with silences in the audio.

    args:
        transcript_file (str): Path to the transcript file.
        audio_file (str): Path to the audio file.
    """
    raise NotImplementedError("Method not implemented.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", type=str, help="Transcript file path.", required=True)
    parser.add_argument("--transcript_type", type=str, help="Transcript type. Options are \"whisper\" or \"datagain\".", required=True)
    parser.add_argument("--audio", type=str, help="Audio file path.", required=True)
    parser.add_argument("--silence_threshold", type=float, help="Silence threshold. Defaults to 0.0.", default=0.0)

    args = parser.parse_args()

    trans_type = args.transcript_type.lower()
    if trans_type == "whisper":
        annotate_silences_whisper(args.transcript, args.audio)
    elif trans_type == "datagain":
        annotate_silences_datagain(args.transcript, args.audio)
    else:
        raise ValueError("Invalid transcript type. Options are \"whisper\" or \"datagain\".")

    print("DONE.")
