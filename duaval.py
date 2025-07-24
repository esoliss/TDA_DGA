import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- Cargar CSV ---
df_raw = pd.read_csv("raw CSV export/U2B.csv", sep=';', encoding='utf-8-sig')

# --- Renombrar columnas ---
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
df_raw = df_raw.rename(columns=column_mapping)

# --- Convertir timestamp y limpiar datos ---
df_raw['Timestamp'] = pd.to_datetime(df_raw['Timestamp'], format='%d/%m/%Y %H:%M:%S')
numeric_cols = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'CO', 'CO2', 'O2', 'GasComb', 'H2O']
for col in numeric_cols:
    df_raw[col] = df_raw[col].astype(str).str.replace(',', '.', regex=False)
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

df_clean = df_raw.dropna(subset=numeric_cols).copy()

# --- Filtrar por año 2025 ---
df_2025 = df_clean[df_clean['Timestamp'].dt.year == 2025].copy()
if df_2025.empty:
    raise ValueError("No hay datos del año 2025.")

# --- FILTRO por rango de fechas ---
fecha_inicio = "2025-01-01"
fecha_fin = "2025-12-31"
mask = (df_2025['Timestamp'] >= fecha_inicio) & (df_2025['Timestamp'] <= fecha_fin)
df_filtrado = df_2025.loc[mask].copy()

if df_filtrado.empty:
    raise ValueError(f"No hay datos entre {fecha_inicio} y {fecha_fin}.")

# --- Graficar en el Triángulo de Duval ---
import duvals_triangle_plotter as dtp

methane_pct = []
acetylene_pct = []
ethylene_pct = []
fechas = []

for _, row in df_filtrado.iterrows():
    m, a, e = row['CH4'], row['C2H2'], row['C2H4']
    total = m + a + e
    if total == 0:
        continue  # evitar división por cero
    methane_pct.append(m / total)
    acetylene_pct.append(a / total)
    ethylene_pct.append(e / total)
    fechas.append(row['Timestamp'].strftime("%Y-%m-%d"))

# Obtener y graficar los puntos
traces = [dtp.get_duval_points_traces([methane_pct[i]], [acetylene_pct[i]], [ethylene_pct[i]], fechas[i])
          for i in range(len(fechas))]

fig = dtp.get_duvals_triangle_plot(traces, show_plot=True)
