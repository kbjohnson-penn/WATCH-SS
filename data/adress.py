import numpy as np
import pandas as pd
import pylangacq
import re
from sklearn.model_selection import train_test_split

def split_train_into_train_dev(dev_size=0.3, num_seeds=100):
    np.random.seed(1234567)

    trn_ids, dev_ids = [], []
    
    cn_trn_ocs = pd.read_csv("/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/dementia_bank/ADReSS-IS2020/train/cc_meta_data.txt", delimiter=";", index_col="ID   ")
    cn_trn_ocs.index = cn_trn_ocs.index.str.strip()
    cn_trn_ocs["AD_dx"] = 0
    ad_trn_ocs = pd.read_csv("/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/dementia_bank/ADReSS-IS2020/train/cd_meta_data.txt", delimiter=";", index_col="ID   ")
    ad_trn_ocs.index = ad_trn_ocs.index.str.strip()
    ad_trn_ocs["AD_dx"] = 1

    trn_ocs = pd.concat([cn_trn_ocs, ad_trn_ocs], axis=0)
    trn_ocs.columns = trn_ocs.columns.str.strip()
    
    metric_scores = []
    seeds = np.random.randint(0, 10000, size=num_seeds)
    for seed in seeds:
        trn_ids, dev_ids = train_test_split(trn_ocs.index.values, test_size=dev_size, stratify=trn_ocs.loc[:, ["AD_dx", "gender"]], random_state=seed)
        age_dist_trn = trn_ocs.loc[trn_ids, "age"].mean()
        age_dist_dev = trn_ocs.loc[dev_ids, "age"].mean()
        metric_scores.append(abs(age_dist_trn - age_dist_dev))

    opt_seed = seeds[np.argmin(metric_scores)]

    return train_test_split(trn_ocs.index.values, test_size=dev_size, stratify=trn_ocs.loc[:, ["AD_dx", "gender"]], random_state=opt_seed)

def load_outcomes():
    # control train
    temp1 = pd.read_csv("/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/dementia_bank/ADReSS-IS2020/train/cc_meta_data.txt", delimiter=";", index_col="ID   ")
    temp1.index = temp1.index.str.strip()
    temp1.columns = temp1.columns.str.strip()
    temp1["gender"] = temp1["gender"].map({" male ": 0, " female ": 1})
    temp1["AD_dx"] = 0
    # dementia train
    temp2 = pd.read_csv("/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/dementia_bank/ADReSS-IS2020/train/cd_meta_data.txt", delimiter=";", index_col="ID   ")
    temp2.index = temp2.index.str.strip()
    temp2.columns = temp2.columns.str.strip()
    temp2["gender"] = temp2["gender"].map({" male ": 0, " female ": 1})
    temp2["AD_dx"] = 1
    # separate train and dev
    trn_ids, dev_ids = split_train_into_train_dev()
    trn_dev = pd.concat([temp1, temp2], axis=0)
    split_idx = ["train" if pt_id in trn_ids else "dev" for pt_id in trn_dev.index.values]
    trn_dev.index = pd.MultiIndex.from_arrays([split_idx, trn_dev.index], names=["split", "ID"])

    # test
    temp3 = pd.read_csv("/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/dementia_bank/ADReSS-IS2020/test/meta_data.txt", delimiter=";", index_col="ID   ")
    temp3.index = temp3.index.str.strip()
    temp3.columns = temp3.columns.str.strip()
    temp3 = temp3.rename(columns={"Label": "AD_dx"})
    temp3.index = pd.MultiIndex.from_arrays([["test"] * temp3.shape[0], temp3.index], names=["split", "ID"])

    # join everything
    lbls = pd.concat([trn_dev, temp3], axis=0)
    lbls["mmse"] = pd.to_numeric(lbls["mmse"], errors="coerce")
    lbls.index.names = ["split", "ID"]

    return lbls.sort_index()

def clean_CHAT_text(text, keep_filler=False):
    text = re.sub(r"\[.*?\]\s*", "", text)                                                  # remove CHAT "[...]" tags
    text = re.sub(r"\+<", "", text)                                                         # remove lazy overlap tags
    text = re.sub(r"<\s*(.*?)\s*>", r"\1", text)                                            # remove CHAT "<...>" tags
    text = re.sub(r"\(\.{1,3}\)", "[silence]", text)                                        # replace pause tags with silence tag
    text = re.sub(r"\([^)]*\)", "", text)                                                   # removing unspoken characters
    text = re.sub(r"xxx", "[inaudible]", text)                                              # replace unintelligible segments with inaudible tag to mimic Datagain
    text = re.sub(r"&=([\w:]+)", lambda m: f"[{m.group(1).replace(':', ' ')}]", text)       # reformat event tags 
    if not keep_filler:
        text = re.sub(r"&(\w+)", r"\1", text)                                               # remove & prefix for Fragments, Fillers, and Nonwords
    text = re.sub(r"\+\S+", "", text)                                                       # remove special utterance terminators
    text = re.sub(r"@\S+", "", text)                                                        # remove special form markers
    text = re.sub(r"([^\s_]+(?:_[^\s_]+)+)", lambda m: m.group(1).replace("_", " "), text)  # split compounds
    text = re.sub(r"‡", "", text)                                                           # remove satellite markers
    text = re.sub(r"(\w+):(\w+)", r"\1\2", text)                                            # remove prolongation markers
    return text

def load_transcripts(annotate_filler=False):
    reader = pylangacq.read_chat("/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/dementia_bank/ADReSS-IS2020/")

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
    if annotate_filler:
        transcripts["Transcript_clean_w_filler"] = transcripts["Transcript"].apply(clean_CHAT_text, keep_filler=True).str.strip()

    # separate train and dev
    trn_ids, dev_ids = split_train_into_train_dev()
    new_split_idx = ["train" if pt_id in trn_ids else "dev" if pt_id in dev_ids else "test" for pt_id in transcripts.index.get_level_values("ID")]
    transcripts.index = pd.MultiIndex.from_arrays([new_split_idx, transcripts.index.get_level_values("ID"), transcripts.index.get_level_values("utt_num")], names=["split", "ID", "utt_num"])

    # labeling
    transcripts["Filler"] = transcripts["Transcript"].str.contains(r"&(?!=)").astype(int)
    transcripts["Repetition"] = transcripts["Transcript"].str.contains(r"\[/\]").astype(int)
    transcripts["Revision"] = transcripts["Transcript"].str.contains(r"\[//\]").astype(int)
    transcripts["Short pause"] = transcripts["Transcript"].str.contains(r"\(\.\)").astype(int)
    transcripts["Medium pause"] = transcripts["Transcript"].str.contains(r"\(\.\.\)").astype(int)
    transcripts["Long pause"] = transcripts["Transcript"].str.contains(r"\(\.\.\.\)").astype(int)
    transcripts["Speech delays"] = (transcripts["Short pause"] | transcripts["Medium pause"] | transcripts["Long pause"]).astype(int)
    transcripts["Vague"] = transcripts["Transcript"].str.contains(r"\[\+ (?:es|cir)\]").astype(int)
    transcripts["Phonological Error"] = transcripts["Transcript"].str.contains(r"\[\*\s+p[^\]]*\]").astype(int)
    transcripts["Semantic Error"] = transcripts["Transcript"].str.contains(r"\[\*\s+s[^\]]*\]").astype(int)
    transcripts["Neologistic Error"] = transcripts["Transcript"].str.contains(r"\[\*\s+n[^\]]*\]").astype(int)
    transcripts["Morphological Error"] = transcripts["Transcript"].str.contains(r"\[\*\s+m[^\]]*\]").astype(int)
    transcripts["Dysfluency"] = transcripts["Transcript"].str.contains(r"\[\*\s+d[^\]]*\]").astype(int)
    transcripts["Substitution Error"] = (transcripts["Phonological Error"] | transcripts["Semantic Error"] | transcripts["Neologistic Error"] | transcripts["Morphological Error"] | transcripts["Dysfluency"]).astype(int)

    columns = ["T_start_ms", "T_end_ms", "Timestamp", "Speaker", "Transcript", "Transcript_clean", "Filler", "Repetition", "Revision", "Short pause", "Medium pause", "Long pause", "Speech delays", "Vague", "Phonological Error", "Semantic Error", "Neologistic Error", "Morphological Error", "Dysfluency", "Substitution Error"]
    
    if annotate_filler:
        columns += ["Transcript_clean_w_filler"]

    return transcripts[columns].sort_index()
