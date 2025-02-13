import argparse
import json
from pydub import AudioSegment
from pydub.silence import detect_silence
from moviepy import VideoFileClip
import ffmpeg
import os
import tempfile
import sys


def process_transcript(transcript_file):
    if not transcript_file.lower().endswith(".json"):
        raise ValueError("This method only supports JSON Whisper transcripts.")

    with open(transcript_file, "r") as file:
        transcript = json.load(file)

    return transcript


def process_audio(audio_file, min_silence_len, silence_thresh):
    if audio_file.lower().endswith(".wav"):
        audio_segment = AudioSegment.from_file(audio_file, format="wav")
    elif audio_file.lower().endswith(".mp4"):
        sys.stdout = open(os.devnull, 'w')
        video = VideoFileClip(audio_file)
        sys.stdout = sys.__stdout__

        temp_dir = tempfile.gettempdir()
        if not os.access(temp_dir, os.W_OK):
            raise PermissionError(f"Write access is not allowed for temp directory: {temp_dir}")
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_dir, delete=False) as temp_audio_file:
                audio = video.audio
                audio.write_audiofile(temp_audio_file.name, codec='pcm_s16le')
                audio_segment = AudioSegment.from_file(temp_audio_file.name, format="wav")

        except Exception as e:
            raise OSError(f"Error processing audio: {str(e)}")
    else:
        raise ValueError("Use a file type that contains audio.")
    
    print("Detecting silences...")
    silences = detect_silence(
        audio_segment,
        min_silence_len=min_silence_len if min_silence_len is not None else 2000, # 2 seconds
        silence_thresh=silence_thresh if silence_thresh is not None else -45 # default silence threshold
    )

    silences = [[start / 1000, end / 1000] for start, end in silences]
    print(f"Silences detected at [start, end] timestamps: {silences}\nTotal number of silences: {len(silences)}")
    return silences


def save_transcript_to_file(transcript, transcript_file, output):
    """
    Save the annotated transcript to a file in the specified directory.

    args:
        transcript (dict): The annotated transcript to be saved.
        transcript_file (str): The path to the original transcript file to extract the name.
        output (str): The directory where the annotated transcript will be saved.
    """
    transcript_filename = os.path.basename(transcript_file)
    file_name = os.path.splitext(transcript_filename)[0]
    output_file = os.path.join(output, f"{file_name}_annotated_silences.json")
    
    with open(output_file, "w") as f:
        json.dump(transcript, f, indent=4)

    print(f"Annotated transcript saved to: {output_file}")


def annotate_silences_crisperwhisper(transcript_file, audio_file, output, min_silence_len, silence_thresh):
    """
    Annotate CrisperWhisper transcript with silences in the audio.

    args:
        transcript_file (str): Path to the transcript file.
        audio_file (str): Path to the audio file.
        min_silence_len (int): Minimum duration of silence (miliseconds).
        silence_thresh (int): Volume threshold for detecting silence.
    """
    transcript = process_transcript(transcript_file)
    silences = process_audio(audio_file, min_silence_len, silence_thresh)

    appended_silences = set()
    i = 0

    while i < len(transcript["chunks"]):
        chunk = transcript["chunks"][i]
        start_timestamp = chunk["timestamp"][0]

        for silence in silences:
            silence_tuple = tuple(silence)
            if silence_tuple in appended_silences:
                continue
            if silence[0] < start_timestamp:
                silence0 = round(silence[0], 2)
                silence1 = round(silence[1], 2)
                transcript["chunks"].insert(
                    i, {"text": "[silence]", "timestamp": [silence0, silence1], "duration": silence1 - silence0}
                )
                appended_silences.add(silence_tuple)
                break
        else:
            i += 1

    save_transcript_to_file(transcript, transcript_file, output)


def annotate_silences_whisper(transcript_file, audio_file, output, min_silence_len, silence_thresh):
    """
    Annotate Whisper transcript with silences in the audio.

    args:
        transcript_file (str): Path to the transcript file.
        audio_file (str): Path to the audio file.
        min_silence_len (int): Minimum duration of silence (miliseconds).
        silence_thresh (int): Volume threshold for detecting silence.
        output (str): Directory where the annotated transcript will be saved.
    """
    transcript = process_transcript(transcript_file)
    silences = process_audio(audio_file, min_silence_len, silence_thresh)

    appended_silences = set()
    i = 0

    while i < len(transcript["word_segments"]):
        word = transcript["word_segments"][i]
        start_timestamp = word["start"]

        for silence in silences:
            silence_tuple = tuple(silence)
            if silence_tuple in appended_silences:
                continue
            if silence[0] < start_timestamp:
                silence0 = round(silence[0], 2)
                silence1 = round(silence[1], 2)
                transcript["word_segments"].insert(
                    i, {"word": "[silence]", "start": silence0, "end": silence1, "duration": silence1 - silence0}
                )
                appended_silences.add(silence_tuple)
                break
        else:
            i += 1

    save_transcript_to_file(transcript, transcript_file, output)


def annotate_silences_datagain(transcript_file, audio_file):
    """
    Annotate Datagain transcript with silences in the audio. ie [silence, n_seconds]

    args:
        transcript_file (str): Path to the transcript file.
        audio_file (str): Path to the audio file.
    """
    raise NotImplementedError("Method not implemented.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", type=str, help="Transcript file path.", required=True)
    parser.add_argument("--transcript_type", type=str, help="Transcript type. Options are [\"crisperwhisper\", \"whisper\", \"datagain\"]", required=True)
    parser.add_argument("--audio", type=str, help="Audio file path.", required=True)
    parser.add_argument("--output", type=str, help="Directory to save the annotated transcript.", default=".", required=False)
    parser.add_argument("--silence_thresh", type=int, help="Silence threshold. Defaults to 16 decibels below the average loudness of the audio segment.", required=False)
    parser.add_argument("--min_silence_len", type=int, help="Minimum duration of silence (miliseconds). Defaults to 2 seconds.", required=False)

    args = parser.parse_args()

    trans_type = args.transcript_type.lower()
    if trans_type == "crisperwhisper":
        annotate_silences_crisperwhisper(args.transcript, args.audio, args.output, args.silence_thresh, args.min_silence_len)
    elif trans_type == "whisper":
        annotate_silences_whisper(args.transcript, args.audio, args.output, args.silence_thresh, args.min_silence_len)
    elif trans_type == "datagain":
        annotate_silences_datagain(args.transcript, args.audio)
    else:
        raise ValueError("Invalid transcript type. Options are [\"crisperwhisper\", \"whisper\", \"datagain\"]")

    print("DONE.")
