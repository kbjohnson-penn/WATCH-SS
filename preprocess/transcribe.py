import argparse
import json
import os
import sys
import whisper

def transcribe(audio_path, output_dir, output_format, whisper_model, language, temperature):
    """
    Transcribe a audio file and write result to file.

    args:
        audio_path (str): Path to audio file
        output_dir (str): Path to output file
        output_format (str): Whisper output format (e.g., tsv, json)
        whisper_model (str): Whisper model to use
        language (str): Language to transcribe to
        temperature (float): Temperature for sampling

    return:
        None
    """
    # Transcribe audio with Whisper
    model = whisper.load_model(whisper_model, device="cuda")
    result = model.transcribe(audio_path, language=language, temperature=temperature, word_timestamps=True)

    # Save the result
    if output_dir is None:
        output_dir = audio_path.rsplit(".", 1)[0]

    results_writer = whisper.utils.get_writer(output_format, output_dir)
    results_writer(result, audio_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, help="Audio file path", required=True)
    parser.add_argument("--output_dir", type=str, help="Output directory. Defaults to the same directory as the input video.", default=None)
    parser.add_argument("--output_format", type=str, help="Whisper output format. Defaults to \"json\".", default="json")
    parser.add_argument("--whisper_model", type=str, help="Whisper model. Defaults to \"large-v3 turbo\".", default="turbo")
    parser.add_argument("--language", type=str, help="Audio language. Defaults to English (\"en\").", default="en")
    parser.add_argument("--temperature", type=float, help="Temperature for scaling. Defaults to 0.0.", default=0.0)

    args = parser.parse_args()

    if os.path.exists(os.path.join(args.output_dir, os.path.basename(args.audio).rsplit(".", 1)[0] + ".json")):
        print(f"Transcript for file {args.audio} already exists in output directory {args.output_dir}.")
    else:
        transcribe(args.audio, args.output_dir, args.output_format, args.whisper_model, args.language, args.temperature)

    print("DONE.")
    