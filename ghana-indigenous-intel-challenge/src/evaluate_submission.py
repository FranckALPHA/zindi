#!/usr/bin/env python3
"""
Calculateur de Score et Analyseur de Soumissions pour Zindi
Métrique Officielle : Macro F1-Score

Utilisations :
1. Évaluer des prédictions contre la vérité terrain :
   python3 src/evaluate_submission.py --true y_true.csv --pred submission.csv

2. Comparer deux fichiers de soumission entre eux (ex: Benchmark vs Nouvelle soumission) :
   python3 src/evaluate_submission.py --compare sub_benchmark.csv sub_new.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

CLASSES = ['HEAVYRAIN', 'MEDIUMRAIN', 'NORAIN', 'SMALLRAIN']

def compute_macro_f1(y_true, y_pred, verbose=True):
    """Calcule le Macro F1 officiel Zindi et le rapport détaillé par classe."""
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    
    macro_f1 = np.mean(f1)
    
    if verbose:
        print("\n" + "=" * 65)
        print("📊 RAPPORT DÉTAILLÉ DU CALCULATEUR DE SCORE (ZINDI OFFICIAL)")
        print("=" * 65)
        print(f"{'Classe':<12} | {'Précision':<10} | {'Rappel':<10} | {'F1-Score':<10} | {'Support'}")
        print("-" * 65)
        for i, c in enumerate(CLASSES):
            print(f"{c:<12} | {p[i]:<10.4f} | {r[i]:<10.4f} | {f1[i]:<10.4f} | {support[i]}")
        print("-" * 65)
        print(f"🎯 SCORE OFFICIEL MACRO F1 : {macro_f1:.6f}")
        print("=" * 65)
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in CLASSES], columns=[f"Pred_{c}" for c in CLASSES])
        print("\nMatrice de Confusion :\n", cm_df)
        print("=" * 65)
        
    return macro_f1, f1

def compare_two_submissions(sub_ref_path, sub_new_path):
    """Analyse chirurgicale des divergences entre deux fichiers de soumission."""
    ref = pd.read_csv(sub_ref_path)
    new = pd.read_csv(sub_new_path)
    
    assert ref.shape[0] == new.shape[0], "Taille différente entre les deux fichiers !"
    
    merged = ref[['ID', 'Target']].merge(new[['ID', 'Target']], on='ID', suffixes=('_ref', '_new'))
    total = len(merged)
    agreements = (merged['Target_ref'] == merged['Target_new']).sum()
    diffs = total - agreements
    agreement_rate = (agreements / total) * 100
    
    print("\n" + "=" * 65)
    print("🔬 COMPARATEUR DE SOUMISSIONS ZINDI")
    print("=" * 65)
    print(f"Fichier Référence : {os.path.basename(sub_ref_path)}")
    print(f"Fichier Comparé   : {os.path.basename(sub_new_path)}")
    print("-" * 65)
    print(f"Observations totales        : {total}")
    print(f"Prédictions identiques      : {agreements} ({agreement_rate:.2f}%)")
    print(f"Prédictions divergentes     : {diffs} ({100 - agreement_rate:.2f}%)")
    print("-" * 65)
    
    print("\nRépartition des classes dans chaque fichier :")
    dist = pd.DataFrame({
        'Référence': ref['Target'].value_counts().reindex(CLASSES).fillna(0).astype(int),
        'Nouvelle': new['Target'].value_counts().reindex(CLASSES).fillna(0).astype(int)
    })
    dist['Différence'] = dist['Nouvelle'] - dist['Référence']
    print(dist)
    
    print("\nMatrice de Transition (Où vont les classes de référence ?) :")
    crosstab = pd.crosstab(merged['Target_ref'], merged['Target_new'], margins=True)
    print(crosstab)
    print("=" * 65)

def main():
    parser = argparse.ArgumentParser(description="Calculateur officiel de Macro F1 pour Zindi")
    parser.add_argument("--compare", nargs=2, metavar=('SUB1', 'SUB2'), help="Comparer deux fichiers CSV de soumission")
    parser.add_argument("--eval", nargs=2, metavar=('TRUE_CSV', 'PRED_CSV'), help="Évaluer un fichier de prédictions contre un fichier réel")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_two_submissions(args.compare[0], args.compare[1])
    elif args.eval:
        df_true = pd.read_csv(args.eval[0])
        df_pred = pd.read_csv(args.eval[1])
        compute_macro_f1(df_true['Target'], df_pred['Target'])
    else:
        # Démo automatique par défaut sur les soumissions existantes
        base_dir = os.path.dirname(os.path.abspath(__file__))
        proj_dir = os.path.abspath(os.path.join(base_dir, ".."))
        sub_dir = os.path.join(proj_dir, "submissions")
        
        bench_file = os.path.join(sub_dir, "submission_starter_benchmark.csv")
        ens_file = os.path.join(sub_dir, "submission_benchmark_5seed_ensemble.csv")
        
        if os.path.exists(bench_file) and os.path.exists(ens_file):
            print("[+] Démonstration du comparateur sur vos soumissions existantes :")
            compare_two_submissions(bench_file, ens_file)
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
