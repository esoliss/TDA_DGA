import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from ripser import ripser
from persim import plot_diagrams, wasserstein
import umap.umap_ as umap

# --- Función para cargar datos de una unidad ---
def cargar_unidad(path_csv, year=2025):

    column_mapping = {
    'Hidrógeno (ppm)': 'H2',
    'Metano (ppm)': 'CH4', 
    'Acetileno (ppm)': 'C2H2',
    'Etileno (ppm)': 'C2H4',
    'Etano (ppm)': 'C2H6',
    'Monóxido de carbono (ppm)': 'CO',
    'Dióxido de carbono (ppm)': 'CO2',
    'Oxígeno (ppm)': 'O2',
    'Gas combustible disuelto total (ppm)': 'GasComb',
    'Agua (ppm)': 'H2O'
    }

    df = pd.read_csv(path_csv, sep=';', encoding='utf-8-sig')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S')
    df = df.rename(columns=column_mapping)
    gas_cols = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'CO', 'CO2', 'O2', 'GasComb', 'H2O']

    for col in gas_cols:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_clean = df.dropna(subset=gas_cols).copy()
    df_filtered = df_clean[df_clean['Timestamp'].dt.year == year].copy()
    X_scaled = StandardScaler().fit_transform(df_filtered[gas_cols])

    return X_scaled, df_filtered

# --- Cargar datos de dos unidades ---
X_U1, df_U1 = cargar_unidad("raw CSV export/U1B.csv")
X_U2, df_U2 = cargar_unidad("raw CSV export/U2B.csv")

# --- Homología persistente ---
tda_U1 = ripser(X_U1, maxdim=2)['dgms']
tda_U2 = ripser(X_U2, maxdim=2)['dgms']

# --- Calcular distancias Wasserstein ---
distances = []
for dim in range(3):
    d1, d2 = tda_U1[dim], tda_U2[dim]
    dist = wasserstein(d1, d2) if len(d1) > 0 and len(d2) > 0 else 0.0
    distances.append(dist)

# --- Diagramas de persistencia ---
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for dim in range(3):
    axs[dim].set_title(f"H{dim} - Diagramas de Persistencia\nWasserstein: {distances[dim]:.4f}")
    plot_diagrams([tda_U1[dim], tda_U2[dim]], labels=["Unidad 1", "Unidad 2"], ax=axs[dim], show=False)

plt.tight_layout()
plt.savefig("tda_diagrams_wasserstein_unidades.png")
plt.show()

# --- Visualización en espacio UMAP ---
umap_model = umap.UMAP(n_components=2, random_state=42)
X_U1_umap = umap_model.fit_transform(X_U1)
X_U2_umap = umap_model.fit_transform(X_U2)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].scatter(X_U1_umap[:, 0], X_U1_umap[:, 1], c='blue', alpha=0.6, label="Unidad 1")
axs[0].set_title("Espacio UMAP - Unidad 1")
axs[0].set_xlabel("UMAP1")
axs[0].set_ylabel("UMAP2")
axs[0].legend()

axs[1].scatter(X_U2_umap[:, 0], X_U2_umap[:, 1], c='red', alpha=0.6, label="Unidad 2")
axs[1].set_title("Espacio UMAP - Unidad 2")
axs[1].set_xlabel("UMAP1")
axs[1].set_ylabel("UMAP2")
axs[1].legend()

plt.tight_layout()
plt.savefig("tda_umap_espacio_unidades.png")
plt.show()

# --- Imprimir distancias Wasserstein ---
for i, d in enumerate(distances):
    print(f"Distancia Wasserstein H{i}: {d:.4f}")
