import os, random, warnings
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from transformers import PatchTSTForPretraining
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

# device and seed
dev = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.empty_cache()

train = "/content/drive/MyDrive/Amini GeoFM Decoding the Field Challenge/dummy_satellite_data.csv"
test = "/content/drive/MyDrive/Amini GeoFM Decoding the Field Challenge/test.csv"
subm_csv = "/content/drive/MyDrive/Amini GeoFM Decoding the Field Challenge/sub_kfold_oof.csv"
model = "AminiTech/fm-v2-28M"
token = ""
n_fold = 5

band = ["red", "nir", "swir16", "swir22", "blue", "green", "rededge1", "rededge2", "rededge3", "nir08"]
n_step = 48

def prepro(df):
    all_data = []
    has_label = 'crop_type' in df.columns
    for uid, g in df.groupby("unique_id"):
        g = g.sort_values("time").reset_index(drop=True)
        g["NDVI"] = (g["nir"] - g["red"]) / (g["nir"] + g["red"] + 1e-6)
        g["NDWI"] = (g["nir"] - g["swir16"]) / (g["nir"] + g["swir16"] + 1e-6)
        old_x = np.arange(len(g))
        new_x = np.linspace(0, len(g) - 1, n_step)

        row = {
            "unique_id": uid,
            "NDVI_mean": g["NDVI"].mean(),
            "NDVI_std": g["NDVI"].std(),
            "NDWI_mean": g["NDWI"].mean(),
            "NDWI_std": g["NDWI"].std(),
        }
        if has_label:
            row["crop_type"] = g["crop_type"].iloc[0]

        spec = {}
        for b in band:
            cs = CubicSpline(old_x, g[b])
            spec[b] = cs(new_x)
        all_data.append((row, spec))
    return all_data

def to_seq(data, has_labels=True):
    seqs, feats, labels, ids = [], [], [], []
    for row, spec in data:
        seq = np.stack([spec[b] for b in band], axis=-1)
        seqs.append(seq)
        ids.append(row["unique_id"])
        feats.append([row["NDVI_mean"], row["NDVI_std"], row["NDWI_mean"], row["NDWI_std"]])
        if has_labels:
            labels.append(row["crop_type"])
    return np.array(seqs), np.array(feats), labels if has_labels else None, ids

class GeoDataset(torch.utils.data.Dataset):
    def __init__(self, x): self.x = torch.from_numpy(x).float()
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return {"values": self.x[i]}

def get_cme(model, loader):
    model.eval()
    all_cls, all_mean = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Embeddings."):
            x = batch["values"].to(dev)
            mask = ~torch.isnan(x)
            out = model.model(past_values=x, past_observed_mask=mask, return_dict=True).last_hidden_state
            cls = out[:, 0, 0, :]
            mean = out[:, :, 1:, :].mean(dim=(1, 2))
            all_cls.append(cls.cpu())
            all_mean.append(mean.cpu())
    cls_all = np.concatenate(all_cls)
    mean_all = np.concatenate(all_mean)
    return np.concatenate([cls_all, mean_all], axis=1)

train_r = pd.read_csv(train)
test_r = pd.read_csv(test)

train_d = prepro(train_r)
test_d = prepro(test_r)

x_train_seq, x_train_stat, y_train, id_train = to_seq(train_d)
x_test_seq, x_test_st, _, id_test = to_seq(test_d, has_labels=False)

train_l = torch.utils.data.DataLoader(GeoDataset(x_train_seq), batch_size=32, shuffle=False)
test_l = torch.utils.data.DataLoader(GeoDataset(x_test_seq), batch_size=32, shuffle=False)

model_amini = PatchTSTForPretraining.from_pretrained(model, token=token).to(dev)
e_train = get_cme(model_amini, train_l)
e_test = get_cme(model_amini, test_l)

x_train_f = np.concatenate([e_train, x_train_stat], axis=1)
x_test_f = np.concatenate([e_test, x_test_st], axis=1)

sca = StandardScaler()
x_train_sca = sca.fit_transform(x_train_f)
x_test_sca = sca.transform(x_test_f)

pca = PCA(n_components=256, random_state=seed)
x_train_pca = pca.fit_transform(x_train_sca)
x_test_pca = pca.transform(x_test_sca)

lbl_enc = LabelEncoder()
y_enc = lbl_enc.fit_transform(y_train)

oof_pr = np.zeros((x_train_pca.shape[0], 3))
test_pr = np.zeros((x_test_pca.shape[0], 3))
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train_pca, y_enc)):
    print(f"\nFold {fold + 1}")
    x_tr, x_val = x_train_pca[tr_idx], x_train_pca[val_idx]
    y_tr, y_val = y_enc[tr_idx], y_enc[val_idx]

    clf_base = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=seed, n_estimators=300)
    clf = CalibratedClassifierCV(clf_base, method="isotonic", cv=3)
    clf.fit(x_tr, y_tr)

    oof_pr[val_idx] = clf.predict_proba(x_val)
    test_pr += clf.predict_proba(x_test_pca) / n_fold

print("\nOOF Log Loss:", log_loss(y_enc, oof_pr))

# make submission
subm = pd.DataFrame(test_pr, columns=lbl_enc.inverse_transform(np.arange(test_pr.shape[1])))
subm = subm[["cocoa", "rubber", "oil"]]
subm.insert(0, "unique_id", id_test)
subm.to_csv(subm_csv, index=False)
print("Saved:", subm_csv)
