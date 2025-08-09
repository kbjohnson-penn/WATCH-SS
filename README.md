# WATCH-SS: Warning Assessment and Alerting Tool for Cognitive Health from Spontaneous Speech

WATCH-SS is a trustworthy and interpretable modular framework for detecting cognitive impairment from a patient's speech sample. The manuscript for WATCH-SS is currently under review. A preprint is available [here](https://www.medrxiv.org/content/10.1101/2025.08.06.25333047v1).

## Contents
```
|- data/							# Code to load and preprocess ADReSS/OBSERVER datasets
|- preprocess/						# 
|- detectors/						# Code for the detectors for CI indicators
|- notebooks/						# Jupyter notebooks for detector development and experiments
|- fig/								# Figure files for Markdown files
|- utils.py							# 
|- requirements.txt					# List of Python dependencies for WATCH-SS
|- supplementary_material.pdf		# PDF version of supplementary material for WATCH-SS manuscript
|- supplementary_material.md		# Markdown version of supplementary material for WATCH-SS manuscript
|- README.md 						# This file
```

<!---
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
-->

## Citation
If you use the code or findings from this work in your research, please cite our paper:

```
@article {pugh2025watchss,
	author = {Pugh, Sydney and Hill, Matthew and Hwang, Sy and Wu, Rachel and Jang, Kuk and Iannone, Stacy L and O{\textquoteright}Connor, Karen and O{\textquoteright}Brien, Kyra and Eaton, Eric and Johnson, Kevin B},
	title = {WATCH-SS: A Trustworthy and Explainable Modular Framework for Detecting Cognitive Impairment from Spontaneous Speech},
	elocation-id = {2025.08.06.25333047},
	year = {2025},
	doi = {10.1101/2025.08.06.25333047},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2025/08/08/2025.08.06.25333047},
	eprint = {https://www.medrxiv.org/content/early/2025/08/08/2025.08.06.25333047.full.pdf},
	journal = {medRxiv}
}
```
