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
    st.subheader("🗺️ Mapa espacial de riesgo")
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf_grid.boundary.plot(ax=ax, linewidth=0.3, color="gray")
    if not gdf_simulados.empty:
        gdf_simulados.plot(ax=ax, color="red", markersize=10, alpha=0.7, label="Eventos simulados")
    gdf.plot(ax=ax, color="black", markersize=5, alpha=0.6, label="Eventos reales")
    plt.legend()
    plt.title(titulo_mapa)
    plt.axis("off")
    st.pyplot(fig)

    # -------------------------------
    # Calendario de eventos simulados
    # -------------------------------
    st.subheader("📅 Días con feminicidios simulados")

    if not gdf_simulados.empty:
        gdf_simulados["dia"] = gdf_simulados["Fecha"].dt.date
        conteo_dias = gdf_simulados["dia"].value_counts().sort_index()

        mes_sim_date = pd.to_datetime(mes_simulacion + "-01")
        year, month = mes_sim_date.year, mes_sim_date.month

        cal = calendar.Calendar(firstweekday=0)
        dias_html = "<table style='width:100%; text-align:center; border-collapse: collapse;'>"
        dias_html += f"<tr><th colspan='7' style='font-size:18px'>{calendar.month_name[month]} {year}</th></tr>"
        dias_html += "<tr>" + "".join(f"<th>{d}</th>" for d in ["L", "M", "X", "J", "V", "S", "D"]) + "</tr>"

        for week in cal.monthdatescalendar(year, month):
            dias_html += "<tr>"
            for day in week:
                if day.month != month:
                    dias_html += "<td style='background-color:#f9f9f9'></td>"
                else:
                    if day in conteo_dias.index:
                        n = conteo_dias[day]
                        color = "#ff9999"
                        texto = f"{day.day}<br><small>{n} evento(s)</small>"
                    else:
                        color = "white"
                        texto = str(day.day)
                    dias_html += f"<td style='padding:8px; border:1px solid #ddd; background-color:{color}'>{texto}</td>"
            dias_html += "</tr>"

        dias_html += "</table>"
        st.markdown(dias_html, unsafe_allow_html=True)
    else:
        st.info("El modelo no generó eventos simulados para este mes.")

    # -------------------------------
    # Descarga de resultados
    # -------------------------------
    if not gdf_simulados.empty:
        def to_geojson_bytes(gdf):
            return gdf.to_json().encode('utf-8')

        def to_shapefile_bytes(gdf):
            with tempfile.TemporaryDirectory() as tmpdir:
                shp_path = os.path.join(tmpdir, "simulados.shp")
                gdf.to_file(shp_path)
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                        file = os.path.join(tmpdir, f"simulados{ext}")
                        if os.path.exists(file):
                            zf.write(file, arcname=f"simulados{ext}")
                return zip_buffer.getvalue()

        geojson_bytes = to_geojson_bytes(gdf_simulados)
        shapefile_bytes = to_shapefile_bytes(gdf_simulados)

        st.download_button("📥 Descargar simulación (GeoJSON)", geojson_bytes, file_name="simulados.geojson", mime="application/geo+json")
        st.download_button("📦 Descargar simulación (Shapefile ZIP)", shapefile_bytes, file_name="simulados.zip", mime="application/zip")
