import re
from pathlib import Path
import pandas as pd

LOG_DIR = Path("../logs/test_logs")
OUTPUT_CSV = Path("../results/test_results.csv")

def extract(pattern, text, cast=None, default = None):
    match = re.search(pattern,text)
    
    if not match:
        return default
    
    value = match.group(1)
    
    if cast:
        return cast(value)
    
    return value

def parse_classification_metrics(text, class_name):
    pattern = (
        rf"{class_name}\s+"
        r"([0-9.]+)\s+"
        r"([0-9.]+)\s+"
        r"([0-9.]+)\s+"
        r"(\d+)"
    )
    
    match = re.search(pattern, text)
    
    if not match:
        return None
    
    return{
        f"{class_name}_precision": float(match.group(1)),
        f"{class_name}_recall": float(match.group(2)),
        f"{class_name}_f1": float(match.group(3)),
        f"{class_name}_support": int(match.group(4))
    }
    
def parse_filename(name):
    info = {}
    
    info["focal"] =  "on" if "focal" in name else "off"
    info["topk"] = "on" if "topk" in name else "off"

    info["augment"] = extract(r"augment-(on|off)", name)
    info["sampler"] = extract(r"sampler-(on|off)", name)
    info["pos_weight"] = extract(r"pos_weight-(on|off)", name)
    
    info["focal_alpha"] = extract(r"focal_loss-([0-9.]+)",
                                  name,
                                  float)    
    info["topk_value"] = extract(r"topk-([0-9.]+)", name, float)
    info["epochs"] = extract(r"e-([0-9]+)", name, int)
    
    return info

rows = []

for log_file in sorted(LOG_DIR.glob("*.log")):
    text = log_file.read_text()
    row={}
    
    row["model_name"]=log_file.stem
    row.update(parse_filename(log_file.stem))
    
    row["compression"] = extract(r"Compression:\s+(on|off)",
                                 text)
    
    row["threshold"] = extract(r"Threshold\s+([0-9.]+)", text, float)
    
    row["balanced_accuracy"] = extract(r"Balanced accuracy:\s+([0-9.]+)", text, float)
    
    row["auc"] = extract(r"AUC:\s+([0-9.]+)", text, float)
    row["mean_fake_probability"] = extract(r"Mean fake probability:\s+([0-9.]+)", text, float)
    
    confusion_matrix_match = re.search(
        r"\[\[\s*(\d+)\s+(\d+)\s*\]\s*\n\s*\[\s*(\d+)\s+(\d+)\s*\]\]"
        ,text
    )
    
    if confusion_matrix_match:
        row["tn"] = int(confusion_matrix_match.group(1))
        row["fp"] = int(confusion_matrix_match.group(2))
        row["fn"] = int(confusion_matrix_match.group(3))
        row["tp"] = int(confusion_matrix_match.group(4))
    
    real_metrics = parse_classification_metrics(text, "real")
    fake_metrics = parse_classification_metrics(text, "fake")
    
    if real_metrics: 
        row.update(real_metrics)
    if fake_metrics: 
        row.update(fake_metrics) 
    
    rows.append(row)
    
    
df = pd.DataFrame(rows)
df = df.sort_values(by="auc", ascending=False)
df.to_csv(OUTPUT_CSV, index=False)
    
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
    
print(df)
    
print(f"\n Saved CSV to: {OUTPUT_CSV}")