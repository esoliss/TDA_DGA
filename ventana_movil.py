import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from ripser import ripser
from persim import wasserstein

# --- Cargar y preparar datos operativos ---
df_raw = pd.read_csv("raw CSV export/U2C.csv", sep=';', encoding='utf-8-sig')
df_raw['Timestamp'] = pd.to_datetime(df_raw['Timestamp'], format='%d/%m/%Y %H:%M:%S')

# Renombrar columnas
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
df = df_raw.rename(columns=column_mapping)
gas_cols = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'CO', 'CO2', 'O2', 'GasComb', 'H2O']

# Reemplazar comas y convertir a numérico
for col in gas_cols:
    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=gas_cols).copy()
df = df.sort_values("Timestamp").reset_index(drop=True)

# --- Calcular TDCG ---
df["TDCG"] = df['CO']

# --- Derivada temporal y escalado ---
df_diff = df[gas_cols].diff().dropna()
df_diff_scaled = StandardScaler().fit_transform(df_diff)

# Alinear timestamps y TDCG
timestamps = df["Timestamp"].iloc[1:].reset_index(drop=True)
tdcg = df["TDCG"].iloc[1:].reset_index(drop=True)

# --- Parámetros de ventana móvil ---
window_size = 30
step = 1
maxdim = 1
wasserstein_distances = []

for i in range(0, len(df_diff_scaled) - 2 * window_size + 1, step):
    X1 = df_diff_scaled[i:i+window_size]
    X2 = df_diff_scaled[i+window_size:i+2*window_size]

    dgm1 = ripser(X1, maxdim=maxdim)['dgms']
    dgm2 = ripser(X2, maxdim=maxdim)['dgms']

    dist = wasserstein(dgm1[0], dgm2[0]) if len(dgm1[0]) > 0 and len(dgm2[0]) > 0 else 0.0
    wasserstein_distances.append(dist)

# --- Graficar en dos paneles verticales ---
t_plot = timestamps[window_size:len(wasserstein_distances)+window_size]
tdcg_normalized = tdcg[window_size:len(wasserstein_distances)+window_size] / tdcg.max()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# Gráfico 1: Distancia Wasserstein
ax1.plot(t_plot, wasserstein_distances, color='tab:blue', linewidth=2)
ax1.set_ylabel("Wasserstein (H0)")
ax1.set_title("Distancia Wasserstein entre ventanas móviles")
ax1.grid(True)

# Gráfico 2: TDCG
#ax2.plot(t_plot, tdcg_normalized, color='tab:orange', linestyle='--', linewidth=2)
#ax2.set_ylabel("TDCG (normalizado)")
#ax2.set_xlabel("Fecha")
#ax2.set_title("Gases combustibles totales (TDCG)")
#ax2.grid(True)
# --- Gráfico 2: Todos los gases normalizados ---
for col in gas_cols:
    normalized_series = df[col].iloc[1:].reset_index(drop=True)
    normalized_series = normalized_series[window_size:len(wasserstein_distances)+window_size] / df[col].max()
    ax2.plot(t_plot, normalized_series, label=col, linewidth=1.5)

ax2.set_ylabel("Gases (normalizados)")
ax2.set_xlabel("Fecha")
ax2.set_title("Gases disueltos normalizados")
ax2.grid(True)
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Leyenda fuera del gráfico para claridad

plt.tight_layout()
plt.savefig("tda_doble_grafico_wasserstein_tdcg.png")
plt.show()
