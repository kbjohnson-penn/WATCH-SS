import argparse
import os
import whisperx

def transcribe(audio_path, model_size, language, temperature, device, output_format, output_dir, batch_size=16, compute_type="float16"):
    """
    Transcribe a audio file with WhisperX and write result to file.
    (WhisperX: https://github.com/m-bain/whisperX/tree/main)

    args:
        audio_path (str): Path to audio file
        model_size (str): Whisper model to use
        language (str): Language to transcribe to
        temperature (float): Temperature for sampling
        device (str) : Device to use
        output_format (str): Whisper output format (e.g., tsv, json)
        output_dir (str): Path to output file
        batch_size (int) : Batch size for inference, defaults to 16
        compute_type (str) : Compute type for inference, defaults to float16

    return:
        None
    """
    # Load Whisper model
    model = whisperx.load_model(model_size, device, compute_type=compute_type)

    # Load the audio
    audio = whisperx.load_audio(audio_path)

    # Transcribe
    result = model.transcribe(audio, batch_size=batch_size, language=language)  #, temperature=temperature)

    # Align timestamps
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # Save the result
    if output_dir is None:
        output_dir = audio_path.rsplit(".", 1)[0]

    results_writer = whisperx.utils.get_writer(output_format, output_dir)
    results_writer(result, audio_path, {})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, help="Audio file path", required=True)
    parser.add_argument("--model_size", type=str, help="Whisper model. Defaults to \"large-v3 turbo\".", default="turbo")
    parser.add_argument("--language", type=str, help="Audio language. Defaults to English (\"en\").", default="en")
    parser.add_argument("--temperature", type=float, help="Temperature for scaling. Defaults to 0.0.", default=0.0)
    parser.add_argument("--device", type=str, help="Device. Defaults to \"cuda\".", default="cuda:0")
    parser.add_argument("--output_format", type=str, help="Whisper output format. Defaults to \"json\".", default="json")
    parser.add_argument("--output_dir", type=str, help="Output directory. Defaults to the same directory as the input video.", default=None)

    args = parser.parse_args()

    transcribe(args.audio, args.model_size, args.language, args.temperature, args.device, args.output_format, args.output_dir)
    print("DONE.")
    