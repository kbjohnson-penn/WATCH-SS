import pandas as pd

def load_labels():
    # Train
    adresso_trn_pts = pd.read_csv("/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/adresso/ADReSSo-IS2021/diagnosis/train/adresso-train-mmse-scores-diagnoses.csv")
    adresso_trn_pts = adresso_trn_pts.rename(columns={"Unnamed: 0": "ID Number", "adressfname": "ID"})
    adresso_trn_pts["AD_dx"] = adresso_trn_pts["dx"].map({"cn": 0, "ad": 1})
    adresso_trn_pts = adresso_trn_pts.set_index("ID")

    # Test
    mmse_scores = pd.read_csv("/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/adresso/ADReSSo-IS2021/diagnosis/test/adresso-test-mmse-scores.csv")
    diagnoses = pd.read_csv("/Volumes/biomedicalinformatics_analytics/dev_lab_johnson/adresso/ADReSSo-IS2021/diagnosis/test/adresso-test-diagnoses.csv")

    adresso_tst_pts = pd.merge(mmse_scores, diagnoses, on="ID", how="inner")
    adresso_tst_pts = adresso_tst_pts.rename(columns={"MMSE": "mmse", "Dx": "dx"})
    adresso_tst_pts["AD_dx"] = adresso_tst_pts["dx"].map({"Control": 0, "ProbableAD": 1})
    adresso_tst_pts = adresso_tst_pts.set_index("ID")

    lbls = pd.concat((adresso_trn_pts[["mmse", "AD_dx"]], adresso_tst_pts[["mmse", "AD_dx"]]), keys=("train", "test"))
    lbls.index.names = ["split", "ID"]

    return lbls
    