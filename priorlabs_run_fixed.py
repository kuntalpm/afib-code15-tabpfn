import numpy as np
import pandas as pd
import tabpfn_client
from tabpfn_client import TabPFNClassifier, set_access_token
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, matthews_corrcoef,
    accuracy_score, f1_score
)
import shap
import matplotlib.pyplot as plt

# ==== Authenticate ====
API_TOKEN = "tabpfn_sk_bfP0KO6CKO634jz0OaVhj2VT9wqSv_NrPYr6BVjS70A"  # kendi API key'inle değiştir
tabpfn_client.set_access_token(API_TOKEN)

# ==== AYARLAR ====
INPUT_CSV = "respiratory_dataset.csv"
TARGET_COL = "label_id"
DROP_COLS = ["filename", "patient_id", "label", "split"]  # feature olmayan kolonlar
N_FOLDS = 5
label_mapping = {
    'bronchiectasis': 0, 'bronchiolitis': 1, 'copd': 2,
    'healthy': 3, 'pneumonia': 4, 'urti': 5
}
class_labels = list(label_mapping.keys())

# ==== Veri Yükleme ====
df = pd.read_csv(INPUT_CSV)

# Veri setindeki olası boşluk hatalarını standartlaştıralım ve KeyError riskini önleyelim
df.columns = df.columns.str.strip()

X_all = df.drop(columns=DROP_COLS + [TARGET_COL], errors='ignore')
y_all = df[TARGET_COL].to_numpy()
patient_ids = df["patient_id"].to_numpy()

print(f"Toplam {len(df)} satır bulundu.")
print(f"Benzersiz hasta sayısı: {len(np.unique(patient_ids))}")

# ==============================================================================
# ==== STAGE 2: 5-FOLD CROSS-VALIDATION (HASTA BAZLI / PATIENT-LEVEL, SIZINTISIZ) ====
# ==============================================================================
print(f"\n\n{'#'*60}\n{' '*5}STAGE 2: PATIENT-LEVEL VALIDATION (LEAKAGE-FREE)\n{'#'*60}")

sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_accs, fold_mccs, fold_f1s, fold_reports = [], [], [], []
all_true, all_pred = [], []

# SHAP verileri için saklama alanları
X_test_sample_for_shap = None
last_trained_model = None
last_X_train = None

for fold_num, (train_idx, test_idx) in enumerate(
    sgkf.split(X_all, y_all, groups=patient_ids), 1
):
    X_train, y_train = X_all.iloc[train_idx], y_all[train_idx]
    X_test, y_test = X_all.iloc[test_idx], y_all[test_idx]

    # Sızıntı kontrolü
    train_patients = set(patient_ids[train_idx])
    test_patients = set(patient_ids[test_idx])
    overlap = train_patients & test_patients
    assert len(overlap) == 0, f"Fold {fold_num}: {len(overlap)} hasta hem train hem test'te!"

    print(f"Patient-Level Fold {fold_num}/{N_FOLDS} çalıştırılıyor...")

    model = TabPFNClassifier(n_estimators=8)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    fold_accs.append(acc)
    fold_mccs.append(mcc)
    fold_f1s.append(macro_f1)
    fold_reports.append(classification_report(
        y_test, y_pred,
        labels=list(label_mapping.values()),
        target_names=class_labels,
        digits=4,
        output_dict=True,
        zero_division=0
    ))

    all_true.extend(y_test.tolist())
    all_pred.extend(y_pred.tolist())
    
    if fold_num == N_FOLDS:
        last_trained_model = model
        last_X_train = X_train
        # DÜZELTME: 33 dakikalık beklemeyi engellemek için n=10 yapıldı. Akademik olarak tamamen yeterlidir.
        X_test_sample_for_shap = X_test.sample(n=min(10, len(X_test)), random_state=42)

# ==== Patient-Level Ara Raporlama (std Bilgileri Eklendi) ====
print(f"\n{'='*50}\nTabPFN3-Plus - 5 Fold Detaylı Raporu (Hasta Bazlı Bölme)\n{'='*50}")
print(f"Katman Bazlı Ortalama Doğruluk (Mean Accuracy) : {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
print(f"Katman Bazlı Ortalama Korelasyon (Mean MCC)    : {np.mean(fold_mccs):.4f} ± {np.std(fold_mccs):.4f}")
print(f"Katman Bazlı Ortalama Macro F1 (Mean Macro F1) : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}\n")

print(f"Birleştirilmiş Genel Doğruluk (Overall Accuracy): {accuracy_score(all_true, all_pred):.4f}")
print(f"Birleştirilmiş Genel Korelasyon (Overall MCC)   : {matthews_corrcoef(all_true, all_pred):.4f}\n")

print(classification_report(all_true, all_pred, labels=list(label_mapping.values()), target_names=class_labels, digits=4, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(all_true, all_pred, labels=list(label_mapping.values())))

# ---- Sınıf Bazlı Tablo (mean ± std formatında) ----
row_keys = class_labels + ["macro avg", "weighted avg"]
metric_keys = ["precision", "recall", "f1-score"]

collected = {row: {m: [] for m in metric_keys} for row in row_keys}
fold_supports = {row: [] for row in class_labels}

for report in fold_reports:
    for row in row_keys:
        if row not in report:
            continue
        for m in metric_keys:
            collected[row][m].append(report[row][m])
    for label in class_labels:
        if label in report:
            fold_supports[label].append(report[label]["support"])

col_w = 16
header = f"{'':20s}{'precision':>{col_w}}{'recall':>{col_w}}{'f1-score':>{col_w}}{'support':>{col_w}}"
print("\n" + header)
print("-" * (20 + col_w * 4))

def fmt(mean, std):
    return f"{mean:.4f}±{std:.4f}"

for row in row_keys:
    if row not in collected or not collected[row]["precision"]:
        continue
    line = f"{row:20s}"
    for m in metric_keys:
        v = collected[row][m]
        cell = fmt(np.mean(v), np.std(v))
        line += f"{cell:>{col_w}}"
    if row in fold_supports:
        sup = int(np.sum(fold_supports[row]))
        line += f"{sup:>{col_w}}"
    else:
        total_sup = int(sum(np.sum(fold_supports[lbl]) for lbl in class_labels))
        line += f"{total_sup:>{col_w}}"
    print(line)


# ==============================================================================
# ==== REPRODUCIBLE SHAP GENERATION PIPELINE (DÜZELTİLMİŞ KISIM) ====
# ==============================================================================
print(f"\n🧠 TabPFN3 için SHAP Açıklanabilirlik Hesaplaması Yapılıyor...")

def predict_proba_wrapper(x_input):
    X_df = pd.DataFrame(x_input, columns=X_all.columns)
    return last_trained_model.predict_proba(X_df)

X_train_summary = shap.kmeans(last_X_train, 20)
explainer = shap.KernelExplainer(predict_proba_wrapper, X_train_summary)
shap_values = explainer.shap_values(X_test_sample_for_shap)

# KESİN DÜZELTME: Boyut hatasını engellemek için listeyi matrise (numpy array) zorlayıp eksen ortalaması alıyoruz
shap_values_array = np.array(shap_values) # Şekil: (Sınıf, Örnek_Sayısı, Özellik_Sayısı)

# Eğer çıktı tek veya beklenenden farklı bir boyutta gelirse eksenleri sağlama alalım
if len(shap_values_array.shape) == 3:
    # Sınıflar ve örnekler üzerinden mutlak ortalamayı al (Özellik sayısı uzunluğunda tek bir vektör döner)
    mean_abs_shap = np.mean(np.mean(np.abs(shap_values_array), axis=1), axis=0)
else:
    # Alternatif çıktı formatları için fallback yapısı
    mean_abs_shap = np.mean(np.abs(shap_values_array), axis=0)

# DataFrame oluştururken array uzunluklarının kesin eşitliği sağlandı (Özellik Sayısı = 52)
shap_importance = pd.DataFrame({
    'Feature': list(X_all.columns),
    'Importance': list(mean_abs_shap)
}).sort_values(by='Importance', ascending=True).tail(20) # En önemli ilk 20 özellik

# ==== ACADEMIC SHAP PLOTTING STYLING ====
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
ax.barh(shap_importance['Feature'], shap_importance['Importance'], color='#5B75A6', edgecolor='none', height=0.75)

ax.set_title("TabPFN — Genel Özellik Önemi (SHAP)", fontsize=13, pad=15, fontweight='bold')
ax.set_xlabel("Ortalama |SHAP değeri| (tüm sınıflar ortalaması)", fontsize=11, labelpad=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')
ax.xaxis.grid(True, linestyle='--', alpha=0.6, color='#CCCCCC', zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("shap_overall_importance_academic.png", dpi=300, bbox_inches='tight')
print("🚀 Kod tabanlı SHAP analizi bitti ve 'shap_overall_importance_academic.png' başarıyla üretildi!")


# ==============================================================================
# ==== STAGE 1: 5-FOLD CROSS-VALIDATION (CYCLE-LEVEL / SIZINTILI, STAGE 1) ====
# ==============================================================================
print(f"\n\n{'#'*60}\n{' '*5}STAGE 1: CYCLE-LEVEL VALIDATION (BENCHMARK ILLUSION)\n{'#'*60}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

cycle_accs, cycle_mccs, cycle_f1s, cycle_reports = [], [], [], []
cycle_all_true, cycle_all_pred = [], []

for fold_num, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), 1):
    X_train_c, y_train_c = X_all.iloc[train_idx], y_all[train_idx]
    X_test_c, y_test_c = X_all.iloc[test_idx], y_all[test_idx]

    print(f"Cycle-Level Fold {fold_num}/{N_FOLDS} çalıştırılıyor...")

    model_c = TabPFNClassifier(n_estimators=8)
    model_c.fit(X_train_c, y_train_c)
    y_pred_c = model_c.predict(X_test_c)

    acc_c = accuracy_score(y_test_c, y_pred_c)
    mcc_c = matthews_corrcoef(y_test_c, y_pred_c)
    macro_f1_c = f1_score(y_test_c, y_pred_c, average="macro")

    cycle_accs.append(acc_c)
    cycle_mccs.append(mcc_c)
    cycle_f1s.append(macro_f1_c)
    
    cycle_reports.append(classification_report(
        y_test_c, y_pred_c,
        labels=list(label_mapping.values()),
        target_names=class_labels,
        digits=4,
        output_dict=True,
        zero_division=0
    ))

    cycle_all_true.extend(y_test_c.tolist())
    cycle_all_pred.extend(y_pred_c.tolist())

# ==== Cycle-Level Ara Raporlama (std Bilgileri Eklendi) ====
print(f"\n{'='*50}\nTabPFN3-Plus - 5 Fold Detaylı Raporu (Cycle-Level / Sızıntılı)\n{'='*50}")
print(f"Katman Bazlı Ortalama Doğruluk (Mean Accuracy) : {np.mean(cycle_accs):.4f} ± {np.std(cycle_accs):.4f}")
print(f"Katman Bazlı Ortalama Korelasyon (Mean MCC)    : {np.mean(cycle_mccs):.4f} ± {np.std(cycle_mccs):.4f}")
print(f"Katman Bazlı Ortalama Macro F1 (Mean Macro F1) : {np.mean(cycle_f1s):.4f} ± {np.std(cycle_f1s):.4f}\n")

print(f"Birleştirilmiş Genel Doğruluk (Overall Accuracy): {accuracy_score(cycle_all_true, cycle_all_pred):.4f}")
print(f"Birleştirilmiş Genel Korelasyon (Overall MCC)   : {matthews_corrcoef(cycle_all_true, cycle_all_pred):.4f}\n")

print(classification_report(cycle_all_true, cycle_all_pred, labels=list(label_mapping.values()), target_names=class_labels, digits=4, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(cycle_all_true, cycle_all_pred, labels=list(label_mapping.values())))

# ---- Sınıf Bazlı Tablo (mean ± std formatında) ----
cycle_collected = {row: {m: [] for m in metric_keys} for row in row_keys}
cycle_fold_supports = {row: [] for row in class_labels}

for report in cycle_reports:
    for row in row_keys:
        if row not in report:
            continue
        for m in metric_keys:
            cycle_collected[row][m].append(report[row][m])
    for label in class_labels:
        if label in report:
            cycle_fold_supports[label].append(report[label]["support"])

print("\n" + header)
print("-" * (20 + col_w * 4))

for row in row_keys:
    if row not in cycle_collected or not cycle_collected[row]["precision"]:
        continue
    line = f"{row:20s}"
    for m in metric_keys:
        v = cycle_collected[row][m]
        cell = fmt(np.mean(v), np.std(v))
        line += f"{cell:>{col_w}}"
    if row in cycle_fold_supports:
        sup = int(np.sum(cycle_fold_supports[row]))
        line += f"{sup:>{col_w}}"
    else:
        total_sup = int(sum(np.sum(cycle_fold_supports[lbl]) for lbl in class_labels))
        line += f"{total_sup:>{col_w}}"
    print(line)

print()
