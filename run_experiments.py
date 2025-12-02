import os
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_utils import load_dataset, load_images
from features import extract_deep_features
from classifiers_lib import get_base_classifiers, get_voting_ensemble, get_stacking_ensemble


# ---------------------------------------------------------
# EXTRAÇÃO DE FEATURES PROFUNDAS (COM CRONÔMETRO)
# ---------------------------------------------------------
def compute_features_for_paths(paths, cache_path=None, resize=(224, 224)):
    if cache_path and os.path.isfile(cache_path):
        print(f"Carregando features em cache de {cache_path}")
        return joblib.load(cache_path)

    imgs = load_images(paths)
    print(f"Extraindo features profundas de {len(imgs)} imagens...")

    t0 = time.time()   # cronômetro da extração

    X_list = [extract_deep_features(img, resize=resize) for img in imgs]
    X = np.vstack(X_list)

    print(f"➡ Tempo para extrair {len(imgs)} imagens: {time.time() - t0:.2f} segundos\n")

    if cache_path:
        joblib.dump(X, cache_path)

    return X


# ---------------------------------------------------------
# CROSS-VALIDATION COM TEMPO POR FOLD
# ---------------------------------------------------------
def run_cv(X, y, classifiers, n_splits=10, results_dir='results'):
    os.makedirs(results_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    records = []

    for name, clf in classifiers:
        f1s = []
        print(f"\n=== Executando Cross-Validation para {name} ===")

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            print(f" • Fold {fold+1}/10")

            t0 = time.time()
            try:
                clf.fit(X_tr, y_tr)
                fold_time = time.time() - t0
                print(f"   ➤ Tempo do fold: {fold_time:.2f} s")

                y_pred = clf.predict(X_te)
                f1 = f1_score(y_te, y_pred, average='weighted') * 100
                f1s.append(f1)
            except Exception as e:
                print(f"   ERRO no fold {fold+1}: {e}")
                f1s.append(0.0)

        rec = {'classifier': name, 'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s)}
        records.append(rec)

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------
# AVALIAÇÃO NA VALIDAÇÃO COM TEMPO POR MODELO
# ---------------------------------------------------------
def evaluate_on_valid(classifiers, X_train, y_train, X_valid, y_valid, labels, results_dir='results'):
    os.makedirs(results_dir, exist_ok=True)
    eval_records = []

    for name, clf in classifiers:
        print(f"\n=== Treinando modelo final {name} ===")

        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        print(f" ➤ Tempo de treino de {name}: {train_time:.2f} s")

        y_pred = clf.predict(X_valid)

        f1 = f1_score(y_valid, y_pred, average='weighted') * 100
        acc = accuracy_score(y_valid, y_pred) * 100

        eval_records.append({'classifier': name, 'f1_valid': f1, 'acc_valid': acc})

        # matriz de confusão com nomes
        cm = confusion_matrix(y_valid, y_pred, labels=np.arange(len(labels)))
        cm_percent = (cm.astype(float) / cm.sum(axis=1, keepdims=True)) * 100
        cm_percent = np.nan_to_num(cm_percent)

        print("\nMatriz de confusão (valores %):")
        print(pd.DataFrame(cm_percent, index=labels, columns=labels))
        print("\n")

        plt.figure(figsize=(8, 6))
        sns.heatmap(pd.DataFrame(cm_percent, index=labels, columns=labels),
                    annot=True, fmt=".1f", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Matriz de confusão (%) - {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'confusion_{name}.png'))
        plt.close()

    return pd.DataFrame(eval_records)


# ---------------------------------------------------------
# MAIN COM TEMPO TOTAL
# ---------------------------------------------------------
def main(args):

    total_start = time.time()  # cronômetro total

    ds = load_dataset(args.dataset_root, subfolders=('Train', 'Valid'))
    train_paths, y_train = ds['Train']
    valid_paths, y_valid = ds['Valid']

    print(f"Amostras Train: {len(train_paths)}  |  Valid: {len(valid_paths)}")

    # EXTRAÇÃO DE FEATURES
    X_train = compute_features_for_paths(train_paths, cache_path=os.path.join('cache', 'X_train.pkl'))
    X_valid = compute_features_for_paths(valid_paths, cache_path=os.path.join('cache', 'X_valid.pkl'))

    # ENCODER DE LABELS
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_valid_enc = le.transform(y_valid)

    classifiers = get_base_classifiers()

    # CROSS-VALIDATION
    cv_df = run_cv(X_train, y_train_enc, classifiers, results_dir=args.results_dir)
    print(cv_df)

    # VALIDAÇÃO FINAL
    valid_df = evaluate_on_valid(classifiers, X_train, y_train_enc, X_valid, y_valid_enc, labels=le.classes_, results_dir=args.results_dir)
    print(valid_df)

    # ENSEMBLES
    voting_hard = get_voting_ensemble(classifiers, voting='hard')
    voting_soft = get_voting_ensemble(classifiers, voting='soft')
    stacking = get_stacking_ensemble(classifiers)

    for name, ens in [('voting_hard', voting_hard), ('voting_soft', voting_soft), ('stacking', stacking)]:
        print(f"\n=== Treinando Ensemble: {name} ===")

        t0 = time.time()
        ens.fit(X_train, y_train_enc)
        train_time = time.time() - t0
        print(f" ➤ Tempo de treino do ensemble {name}: {train_time:.2f} s")

        y_pred = ens.predict(X_valid)
        f1 = f1_score(y_valid_enc, y_pred, average='weighted')
        acc = accuracy_score(y_valid_enc, y_pred)
        print(f" ➤ Resultado: f1={f1:.4f} | acc={acc:.4f}")

    # TEMPO TOTAL
    total = time.time() - total_start
    print("\n====================================")
    print(f"Tempo TOTAL do experimento: {total:.2f} segundos")
    print(f"Ou: {total/60:.2f} minutos")
    print("====================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=str, default='source/simpsons')
    parser.add_argument('--results-dir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs('cache', exist_ok=True)
    main(args)
