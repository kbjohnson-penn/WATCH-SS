import os
import pandas as pd
import re
from glob import glob

def load_visits():
    visits = pd.read_csv("/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/swimcap/Penn OBSERVER/note_visit_mapping.csv", header=None)
    visits = visits.rename(columns={0: "visit_id", 1: "visit"})
    visits["visit"] = visits["visit"].apply(lambda x: re.sub(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", lambda m: f"{int(m.group(1)):02d}.{int(m.group(2)):02d}.{m.group(3)}", x))
    visits["provider_id"] = visits["visit"].apply(lambda x: re.search(r"PR(\d+)", x).group(1)).astype(int)
    visits["patient_id"] = visits["visit"].apply(lambda x: re.search(r"PT(\d+)", x).group(1)).astype(int)
    visits["date"] = pd.to_datetime(visits["visit"].apply(lambda x: re.search(r"(\d{2}\.\d{2}\.\d{4})", x).group(1)))
    return visits

def load_transcripts_from_visits(visits):
    trans_dir_fmt = "/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/clinic/{}/transcript/"

    idxs, transcripts = [], []
    for i, row in visits.iterrows():
        try:
            transcript_files = os.listdir(trans_dir_fmt.format(row.visit))
        except FileNotFoundError as e:
            print(f"No transcript directory found for visit {row.visit}")
            continue         

        for f in transcript_files:
            if f.endswith(".xlsx"):  # Datagain transcripts are excel files
                t = pd.read_excel(os.path.join(trans_dir_fmt.format(row.visit), f), engine="openpyxl")
                t["Utterance"] = t.apply(lambda x: f"{x.Timestamp} {x.Speaker}: {x.Transcript}", axis=1)
                transcripts.append( t[["Timestamp", "Speaker", "Transcript", "Utterance"]] )
                idxs.append( f )

    transcripts = pd.concat(transcripts, keys=idxs, names=["visit_file", "line_num"])
    return transcripts

def load_visit_transcript(provider_id, patient_id, date):
    t_files = glob(f"/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/clinic/PR{provider_id}_PT{patient_id}_{date}/transcript/*.xlsx")

    trans = []
    for file in t_files:
        t = pd.read_excel(
            file, 
            engine="openpyxl",
            usecols=["Timestamp", "Speaker", "Transcript"]
        )
    transcript["Text"] = transcript.apply(lambda x: f"{x.Timestamp} {x.Speaker}: {x.Transcript}", axis=1)
    return transcript

def load_penn_transcripts():
    clinic_data_dir = "/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/clinic"

    idxs, transcripts = [], []
    for root, dirs, files in os.walk(clinic_data_dir):
        for f in files:
            if f.endswith(".xlsx"):
                t = pd.read_excel(os.path.join(root, f), engine="openpyxl")
                t["Utterance"] = t.apply(lambda x: f"{x.Timestamp} {x.Speaker}: {x.Transcript}", axis=1)
                transcripts.append( t[["Timestamp", "Speaker", "Transcript", "Utterance"]] )
                idxs.append( f )

    transcripts = pd.concat(transcripts, keys=idxs, names=["visit_file", "line_num"])

    # new index
    pattern = r'(PR\d+)_(PT\d+)_(\d{2}\.\d{2}\.\d{4})'
    new_idx = transcripts.index.get_level_values("visit_file").str.extract(pattern)
    new_idx.columns = ["provider_id", "patient_id", "date"]
    new_idx["date"] = pd.to_datetime(new_idx["date"])
    new_idx["line_num"] = transcripts.index.get_level_values("line_num")
    new_idx = pd.MultiIndex.from_frame(new_idx)
    transcripts.index = new_idx

    return transcripts

def load_penn_cogtst_scores():
    lbls = pd.read_excel("/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/swimcap/Penn OBSERVER/cognitive_test_scores.xlsx")
    return lbls

def load_penn_outcomes():
    outcomes = pd.read_excel("/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/watch/penn_AD_labels.xlsx")
    outcomes["date"] = pd.to_datetime(outcomes["date"])
    outcomes = outcomes.set_index(["provider_id", "patient_id", "date"])
    return outcomes

def load():
    visits = load_visits()
    # transcripts = load_transcripts()
    transcripts = load_transcripts_from_visits(visits)
    return visits, transcripts