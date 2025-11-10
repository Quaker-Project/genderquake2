# app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import tempfile
import zipfile
import io
import matplotlib.pyplot as plt
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from dateutil.parser import parse
import calendar

from simulador import entrenar_modelo, simular_eventos

st.title("🔮 Simulación de riesgo espacial y temporal de feminicidios")

# -------------------------------
# Carga de archivos
# -------------------------------
ruta_robos = st.file_uploader("Sube shapefile de eventos (.zip con .shp, .dbf, .shx...)", type=["zip"])
ruta_contorno = st.file_uploader("Opcional: shapefile de contorno/calles (.zip)", type=["zip"])

cell_size = st.number_input("Tamaño celda rejilla (m)", min_value=100, max_value=2000, value=500, step=100)
umbral = st.slider("Umbral de probabilidad de riesgo", 0.0, 1.0, 0.7, 0.05)
mes_simulacion = st.text_input("Mes a simular (YYYY-MM)", "2019-09")
fecha_entreno_inicio = st.text_input("Fecha inicio entrenamiento (YYYY-MM)", "2017-01")
fecha_entreno_fin = st.text_input("Fecha fin entrenamiento (YYYY-MM)", "2019-08")
titulo_mapa = st.text_input("Título del mapa", f"Riesgo predicho vs eventos reales - {mes_simulacion}")

# -------------------------------
# Funciones auxiliares
# -------------------------------
def cargar_shapefile_zip(zip_file):
    if zip_file is None:
        return None
    with tempfile.TemporaryDirectory() as tmpdir:
        z = zipfile.ZipFile(zip_file)
        z.extractall(tmpdir)
        shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
        if not shp_files:
            st.error("No se encontró archivo .shp en el ZIP.")
            return None
        return gpd.read_file(os.path.join(tmpdir, shp_files[0]))


def parse_fecha_segura(fecha):
    try:
        return parse(str(fecha), dayfirst=True, fuzzy=True)
    except:
        return pd.NaT


if ruta_robos is None:
    st.warning("Sube el shapefile de puntos para continuar.")
    st.stop()

gdf = cargar_shapefile_zip(ruta_robos)
if gdf is None:
    st.stop()

# CRS y fechas
gdf = gdf.to_crs(epsg=4326)
gdf["Fecha"] = gdf["Fecha"].apply(parse_fecha_segura)
gdf = gdf.dropna(subset=["Fecha"])
gdf["month"] = gdf["Fecha"].dt.to_period("M")

# -------------------------------
# Grid espacial
# -------------------------------
xmin, ymin, xmax, ymax = gdf.to_crs(epsg=32616).total_bounds
cols = list(np.arange(xmin, xmax, cell_size))
rows = list(np.arange(ymin, ymax, cell_size))
polygons = []
cell_ids = []
for i, x in enumerate(cols):
    for j, y in enumerate(rows):
        poly = box(x, y, x + cell_size, y + cell_size)
        polygons.append(poly)
        cell_ids.append(f"{i}_{j}")

gdf_grid = gpd.GeoDataFrame({'cell_id': cell_ids}, geometry=polygons, crs="EPSG:32616")
gdf_grid["X"] = gdf_grid.geometry.centroid.x
gdf_grid["Y"] = gdf_grid.geometry.centroid.y

# -------------------------------
# Entrenamiento y simulación
# -------------------------------
mes_sim = pd.Period(mes_simulacion, freq="M")
if st.button("🚀 Ejecutar simulación y predicción"):
    st.info("Entrenando modelo Hawkes y simulando eventos...")

    # --- Entrenar modelo Hawkes ---
    gdf_boundaries = gdf_grid.to_crs(epsg=4326)
    model, gdf_train, gdf_test, t0 = entrenar_modelo(
        gdf_events=gdf.to_crs(epsg=4326),
        gdf_boundaries=gdf_boundaries,
        fecha_inicio=pd.to_datetime(fecha_entreno_inicio),
        fecha_split=pd.to_datetime(fecha_entreno_fin)
    )

    # --- Simular eventos ---
    gdf_simulados = simular_eventos(model)

    # -------------------------------
    # Mapa espacial
    # -------------------------------
    st.s
