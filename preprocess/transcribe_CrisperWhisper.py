import os
import argparse
import sys
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import whisper

HF_TOKEN = "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"

def init_CrisperWhisper():
    '''
    Initialize CrisperWhisper with HuggingFace pipeline.
    (CrisperWhisper: https://github.com/nyrahealth/CrisperWhisper)
    '''
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "nyrahealth/CrisperWhisper"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True, 
        token=HF_TOKEN
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(
        model_id,
        token=HF_TOKEN
    )

    cwhisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps="word",
        torch_dtype=torch_dtype,
        device=device,
    )

    return cwhisper_pipe

def adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.12):
    """
    Adjust pause timings by distributing pauses up to the threshold evenly between adjacent words.

    Copied from https://github.com/nyrahealth/CrisperWhisper/blob/main/utils.py
    """

    adjusted_chunks = pipeline_output["chunks"].copy()

    for i in range(len(adjusted_chunks) - 1):
        current_chunk = adjusted_chunks[i]
        next_chunk = adjusted_chunks[i + 1]

        current_start, current_end = current_chunk["timestamp"]
        next_start, next_end = next_chunk["timestamp"]
        pause_duration = next_start - current_end

        if pause_duration > 0:
            if pause_duration > split_threshold:
                distribute = split_threshold / 2
            else:
                distribute = pause_duration / 2

            # Adjust current chunk end time
            adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)

            # Adjust next chunk start time
            adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)
    pipeline_output["chunks"] = adjusted_chunks

    return pipeline_output

def transcribe(audio_file, chunk_size, output_dir, output_fmt):
    """
    Transcribe a audio file with CrisperWhisper and write result to file.

    args:
        audio_file (str): Path to audio file
        chunk_size (int) : Chunk size in milliseconds
        output_dir (str): Path to output file
        output_format (str): Whisper output format (e.g., tsv, json)

    return:
        None
    """
    # Initialize CrisperWhisper
    cwhisper_pipe = init_CrisperWhisper()

    # Load the audio
    audio = whisper.load_audio(audio_file)

    # Transcribe audio in chunks (necessary to fit within A100's memory)
    result = None
    audio_chunks = np.array_split(audio, audio.shape[0] // chunk_size)
    for i, a_chunk in enumerate(audio_chunks):
        print("Chunk size:", a_chunk.shape)
        hf_pipeline_output = cwhisper_pipe(a_chunk)
        chunk_result = adjust_pauses_for_hf_pipeline_output(hf_pipeline_output)

        if result is None:
            result = chunk_result
        else:
            result["text"] = result["text"] + " " + chunk_result["text"]
            result["chunks"] = result["chunks"] + chunk_result["chunks"]

    # Save the result 
    results_writer = whisper.utils.get_writer(output_fmt, output_dir)
    results_writer(result, audio_file, {})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_file", type=str, default="/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/neuro_visits/datagain_transcripts/PR007_PT054_05.31.2024_ClinicVisit.xlsx")
    parser.add_argument("--chunk_size_ms", type=int, default=1.92e6)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_format", type=str, default="json")
    parser.add_argument("-f")

    args = parser.parse_args()

    transcribe(args.audio_file, args.chunk_size_ms, args.output_dir, args.output_format)
    print("DONE.")
