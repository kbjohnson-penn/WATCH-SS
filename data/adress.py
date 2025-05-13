import pandas as pd
import pylangacq
import re

def clean_CHAT_text(text):
    text = re.sub(r"\[.*?\]\s*", "", text)                                                     # remove CHAT "[...]" tags
    text = re.sub(r"<\s*(.*?)\s*>", r"\1", text)                                            # remove CHAT "<...>" tags
    text = re.sub(r"\(\.{1,3}\)", "[silence]", text)                                        # replace pause tags with ellipses
    text = re.sub(r"\([^)]*\)", "", text)                                                   # removing unspoken characters
    text = re.sub(r"xxx", "[inaudible]", text)                                              # replace unintelligible segment tags with inaudible tag to mimic Datagain
    text = re.sub(r"&=([\w:]+)", lambda m: f"[{m.group(1).replace(':', ' ')}]", text)       # reformat event tags 
    text = re.sub(r"&(\w+)", r"\1", text)                                                   # remove & prefix for Fragments, Fillers, and Nonwords
    text = re.sub(r"\+\S+", "", text)                                                       # remove special utterance terminators
    text = re.sub(r"@\S+", "", text)                                                        # remove special form markers
    text = re.sub(r"([^\s_]+(?:_[^\s_]+)+)", lambda m: m.group(1).replace("_", " "), text)  # split compounds
    text = re.sub(r"‡", "", text)                                                           # remove satellite markers
    return text

def load_CHAT_transcripts():
    reader = pylangacq.read_chat("/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/adresso/ADReSS-IS2020/")

    idxs, transcripts = [], []
    for file, f_utterances in zip(reader.file_paths(), reader.utterances(by_files=True)):
        idxs.append( re.search(r"/ADReSS-IS2020/(train|test)/transcription/(?:[a-z]{2}/)?(S\d+)\.cha", file).groups() )

        data = []
        for u in f_utterances:
            text, timestamps = u.tiers[u.participant].split("\x15", maxsplit=1)
            t_start, t_end = timestamps.split("_", maxsplit=1)
            t_end = t_end.strip("\x15")
            data.append((int(t_start), int(t_end), u.participant, text))

        transcripts.append( pd.DataFrame(data, columns=["T_start_ms", "T_end_ms", "Speaker", "Transcript"]) )

    # post processing
    transcripts = pd.concat(transcripts, keys=idxs, names=["split", "ID", "utt_num"])
    transcripts["Timestamp"] = transcripts["T_start_ms"].apply(lambda x: f"{int((x / 1000) // 3600):02}:{int(((x / 1000) % 3600) // 60):02}:{int((x / 1000) % 60):02}")
    transcripts["Speaker"] = transcripts["Speaker"].map({"PAR": "Patient", "INV": "Provider"})
    transcripts["Transcript"] = transcripts["Transcript"].str.strip()
    transcripts["Transcript_clean"] = transcripts["Transcript"].apply(clean_CHAT_text).str.strip()

    # labeling
    transcripts["Filler speech"] = transcripts["Transcript"].str.contains(r"&\w+").astype(int)
    transcripts["Repetitive speech"] = transcripts["Transcript"].str.contains(r"\[\/\]|\u21AB").astype(int)
    transcripts["Speech delays"] = transcripts["Transcript"].str.contains(r"\(\.{1,3}\)|\^").astype(int)
    transcripts["Paraphasic speech"] = transcripts["Transcript"].str.contains(r"\[\* [A-Za-z0-9:=\-\']+\]|\[//\]").astype(int)
    transcripts["Vague speech"] = transcripts["Transcript"].str.contains(r"\[\+ (?:jar|es|cir)\]").astype(int)

    return transcripts[["T_start_ms", "T_end_ms", "Timestamp", "Speaker", "Transcript", "Transcript_clean", "Filler speech", "Repetitive speech", "Speech delays", "Vague speech", "Paraphasic speech"]]

def load_labels():
    # control train
    temp1 = pd.read_csv("/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/adresso/ADReSS-IS2020/train/cc_meta_data.txt", delimiter=";", index_col="ID   ")
    temp1.columns = temp1.columns.str.strip()
    temp1["gender"] = temp1["gender"].map({" male ": 0, " female ": 1})
    temp1["Label"] = 0
    # dementia train
    temp2 = pd.read_csv("/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/adresso/ADReSS-IS2020/train/cd_meta_data.txt", delimiter=";", index_col="ID   ")
    temp2.columns = temp2.columns.str.strip()
    temp2["gender"] = temp2["gender"].map({" male ": 0, " female ": 1})
    temp2["Label"] = 1
    # test
    temp3 = pd.read_csv("/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/adresso/ADReSS-IS2020/test/meta_data.txt", delimiter=";", index_col="ID   ")
    temp3.columns = temp3.columns.str.strip()

    lbls = pd.concat([temp1, temp2, temp3], axis=0, keys=["train", "train", "test"])
    lbls["mmse"] = pd.to_numeric(lbls["mmse"], errors="coerce")
    lbls.index.names = ["split", "ID"]

    return lbls
