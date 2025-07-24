import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import kmapper as km

# --- Cargar y preparar datos ---
df_raw = pd.read_csv("raw CSV export/U1B.csv", sep=';', encoding='utf-8-sig')
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

# Limpiar y convertir a numérico
gas_cols = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'CO', 'CO2', 'O2', 'GasComb', 'H2O']
for col in gas_cols:
    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')
df_clean = df.dropna(subset=gas_cols).copy()

# Filtrar año 2025
#df_2025 = df_clean[df_clean['Timestamp'].dt.year == 2025].copy()
df_2025 = df_clean

if df_2025.empty:
    raise ValueError("No hay datos del año 2025.")

# --- Normalizar los datos ---
X = df_2025[gas_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Transponer datos


# --- Aplicar Mapper ---
mapper = km.KeplerMapper(verbose=1)

# Lente: primer componente principal
lens = mapper.fit_transform(X_scaled, projection=PCA(n_components=1))

# Crear el grafo
graph = mapper.map(lens,
                   X_scaled,
                   cover=km.Cover(n_cubes=10, perc_overlap=0.5),
                   clusterer=km.cluster.DBSCAN(eps=0.5, min_samples=3))

# --- Visualizar el grafo ---
mapper.visualize(graph,
                 path_html="mapper_output.html",
                 title="Análisis TDA con Mapper - Gases en Transformador 2025")
