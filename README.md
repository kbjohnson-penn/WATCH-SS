# WATCH-SS

NOTE: Need to change the name to something more general on detecting diagnostic clues for cognitive impairment from language.


## Annotate Silences in Transcripts
The `annotate_silences.py` Python script allows you to annotate silence periods in audio files and add these annotations to corresponding transcripts. It currently supports Whisper and CrisperWhisper transcripts.

### Features
- Detects silences in audio files (MP4, WAV) and annotates transcript with silence intervals.
- Supports JSON transcripts from Whisper and CrisperWhisper models.
- Option to specify minimum silence duration and silence threshold for better accuracy.

You can run the script directly from the command line using the following syntax:

```bash
python annotate_silences.py --transcript <transcript_file> --transcript_type <transcript_type> --audio <audio_file> --output <output_directory> [--silence_thresh <threshold>] [--min_silence_len <duration>]
```

### Arguments
- --transcript (required): Path to the transcript file (JSON format).
- --transcript_type (required): Type of transcript. Options are: crisperwhisper, whisper
- --audio (required): Path to the audio or video file (MP4 or WAV).
- --output (optional): Directory where the annotated transcript will be saved. Defaults to the current directory.
- --silence_thresh (optional): Silence threshold in decibels. Default is -45 dB.
- --min_silence_len (optional): Minimum silence length in milliseconds. Default is 2000 ms (2 seconds).
