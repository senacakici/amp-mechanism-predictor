"""
features.py
-----------
Extracts physicochemical and compositional features from AMP sequences.

Feature sets:
  - Amino acid composition (AAC)
  - Dipeptide composition (DPC) — optional, high-dimensional
  - Physicochemical descriptors: charge, hydrophobicity, amphipathicity,
    isoelectric point, molecular weight, instability index

Reference scales from literature (Kyte-Doolittle, Eisenberg, etc.)
"""

import numpy as np
import pandas as pd
from typing import List, Dict

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Net charge at physiological pH (simplified)
CHARGE = {
    "A": 0, "C": 0, "D": -1, "E": -1, "F": 0,
    "G": 0, "H": 0.1, "I": 0, "K": 1, "L": 0,
    "M": 0, "N": 0, "P": 0, "Q": 0, "R": 1,
    "S": 0, "T": 0, "V": 0, "W": 0, "Y": 0,
}

# Molecular weight (Da) per residue
MW = {
    "A": 89.09, "C": 121.15, "D": 133.10, "E": 147.13, "F": 165.19,
    "G": 75.03, "H": 155.16, "I": 131.17, "K": 146.19, "L": 131.17,
    "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
    "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
}


def amino_acid_composition(sequence: str) -> Dict[str, float]:
    """20-dimensional normalized amino acid composition."""
    seq = sequence.upper()
    n = len(seq)
    return {f"AAC_{aa}": seq.count(aa) / n for aa in AMINO_ACIDS}


def physicochemical_features(sequence: str) -> Dict[str, float]:
    """Key physicochemical descriptors relevant to AMP activity."""
    seq = sequence.upper()
    n = len(seq)

    net_charge = sum(CHARGE.get(aa, 0) for aa in seq)
    hydro_values = [HYDROPHOBICITY.get(aa, 0) for aa in seq]
    mean_hydro = np.mean(hydro_values)
    hydro_moment = _calc_hydrophobic_moment(seq)
    mol_weight = sum(MW.get(aa, 110) for aa in seq) - 18.02 * (n - 1)

    # Fraction of specific residue types
    frac_cationic = (seq.count("K") + seq.count("R")) / n
    frac_anionic = (seq.count("D") + seq.count("E")) / n
    frac_hydrophobic = sum(1 for aa in seq if aa in "FILMVWY") / n
    frac_aromatic = sum(1 for aa in seq if aa in "FWY") / n
    frac_proline = seq.count("P") / n
    frac_glycine = seq.count("G") / n

    return {
        "length": n,
        "net_charge": net_charge,
        "mean_hydrophobicity": mean_hydro,
        "hydrophobic_moment": hydro_moment,
        "molecular_weight": mol_weight,
        "frac_cationic": frac_cationic,
        "frac_anionic": frac_anionic,
        "frac_hydrophobic": frac_hydrophobic,
        "frac_aromatic": frac_aromatic,
        "frac_proline": frac_proline,
        "frac_glycine": frac_glycine,
        "charge_hydro_ratio": net_charge / (mean_hydro + 1e-6),
    }


def _calc_hydrophobic_moment(sequence: str, angle: float = 100.0) -> float:
    """
    Eisenberg hydrophobic moment — measure of amphipathicity.
    Assumes alpha-helical periodicity (100 deg rotation per residue).
    """
    seq = sequence.upper()
    n = len(seq)
    if n == 0:
        return 0.0
    
    angle_rad = np.radians(angle)
    sum_sin = sum(
        HYDROPHOBICITY.get(aa, 0) * np.sin(i * angle_rad)
        for i, aa in enumerate(seq)
    )
    sum_cos = sum(
        HYDROPHOBICITY.get(aa, 0) * np.cos(i * angle_rad)
        for i, aa in enumerate(seq)
    )
    return np.sqrt(sum_sin**2 + sum_cos**2) / n


def extract_all_features(sequence: str, include_dpc: bool = False) -> Dict[str, float]:
    """
    Extract full feature vector for a single peptide sequence.
    
    Args:
        sequence: Amino acid sequence (single-letter code)
        include_dpc: Include dipeptide composition (adds 400 features)
    
    Returns:
        Dict mapping feature names to values
    """
    features = {}
    features.update(physicochemical_features(sequence))
    features.update(amino_acid_composition(sequence))
    if include_dpc:
        features.update(_dipeptide_composition(sequence))
    return features


def _dipeptide_composition(sequence: str) -> Dict[str, float]:
    """400-dimensional dipeptide composition."""
    seq = sequence.upper()
    n = len(seq) - 1
    if n <= 0:
        return {f"DPC_{a}{b}": 0.0 for a in AMINO_ACIDS for b in AMINO_ACIDS}
    counts = {}
    for i in range(n):
        dp = seq[i:i+2]
        if all(c in AMINO_ACIDS for c in dp):
            counts[dp] = counts.get(dp, 0) + 1
    return {
        f"DPC_{a}{b}": counts.get(f"{a}{b}", 0) / n
        for a in AMINO_ACIDS for b in AMINO_ACIDS
    }


def featurize_dataframe(
    df: pd.DataFrame,
    seq_col: str = "sequence",
    include_dpc: bool = False,
) -> pd.DataFrame:
    """
    Featurize all sequences in a DataFrame.
    
    Args:
        df: DataFrame with a sequence column
        seq_col: Name of the sequence column
        include_dpc: Whether to include dipeptide composition
    
    Returns:
        DataFrame of features (one row per peptide)
    """
    features_list = [
        extract_all_features(seq, include_dpc=include_dpc)
        for seq in df[seq_col]
    ]
    return pd.DataFrame(features_list, index=df.index)


if __name__ == "__main__":
    # Quick test
    test_seqs = [
        ("magainin_2",   "GIGKFLKSAKKFGKAFVGEIMNS"),
        ("buforin_II",   "TRSSRAGLQFPVGRIHRHLKSRTTSHGR"),
        ("LL37",         "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"),
    ]
    
    print("Feature extraction test\n" + "="*50)
    for name, seq in test_seqs:
        feats = physicochemical_features(seq)
        print(f"\n{name} ({len(seq)} aa)")
        for k, v in feats.items():
            print(f"  {k:<30} {v:.3f}")
