import streamlit as st
import pandas as pd
from simulador import (
    entrenar_y_ajustar_modelo,
    forecast_period_single_sim,
    generar_calendario_eventos,
    obtener_fechas_simuladas
)

st.set_page_config(page_title="Simulador de Feminicidios", layout="wide")
st.title("🔮 Simulador de feminicidios con procesos de Hawkes")

uploaded_file = st.file_uploader("📤 Carga un archivo Excel con columna 'Fecha'", type=['xlsx'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    if 'Fecha' not in df.columns:
        st.error("❌ El archivo no contiene una columna llamada 'Fecha'.")
    else:
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        fecha_entreno_inicio = st.date_input("📅 Fecha de inicio del entrenamiento", df['Fecha'].min())
        fecha_entreno_fin = st.date_input("📅 Fecha de fin del entrenamiento", df['Fecha'].max())

        df_train = df[
            (df['Fecha'] >= pd.to_datetime(fecha_entreno_inicio)) &
            (df['Fecha'] <= pd.to_datetime(fecha_entreno_fin))
        ]

        if st.button("🚀 Entrenar modelo"):
            with st.spinner("Entrenando modelo de Hawkes..."):
                t0_train, mu_interp_base, alpha_interp, decay_fit, T_train = entrenar_y_ajustar_modelo(df_train)
                st.session_state['modelo_entrenado'] = (t0_train, mu_interp_base, alpha_interp, decay_fit, T_train)
                st.success("✅ Modelo entrenado correctamente")

        if 'modelo_entrenado' in st.session_state:
            t0_train, mu_interp_base, alpha_interp, decay_fit, T_train = st.session_state['modelo_entrenado']

            fecha_sim_inicio = st.date_input(
                "🧪 Fecha de inicio de simulación",
                value=(fecha_entreno_fin + pd.Timedelta(days=1))
            )
            fecha_sim_fin = st.date_input(
                "🧪 Fecha de fin de simulación",
                value=(fecha_sim_inicio + pd.Timedelta(days=365))
            )

            mu_boost = st.slider("🔥 Multiplicador de intensidad base (mu_boost)", 0.1, 3.0, 1.0, step=0.1)

            if st.button("🎲 Simular eventos futuros"):
                with st.spinner("Simulando eventos..."):
                    events_sim = forecast_period_single_sim(
                        t0_train,
                        mu_interp_base,
                        alpha_interp,
                        decay_fit,
                        pd.to_datetime(fecha_sim_inicio),
                        pd.to_datetime(fecha_sim_fin),
                        mu_boost=mu_boost,
                        T_train=T_train
                    )
                    st.session_state['simulacion'] = (events_sim, fecha_sim_inicio, fecha_sim_fin)
                    st.success(f"✅ Simulación completada. Número de eventos previstos: {len(events_sim)}")

        if 'simulacion' in st.session_state:
            events_sim, fecha_sim_inicio, fecha_sim_fin = st.session_state['simulacion']

            df_real_post_entreno = df[
                df['Fecha'].between(
                    pd.to_datetime(fecha_sim_inicio),
                    pd.to_datetime(fecha_sim_fin)
                )
            ]
            events_real = (df_real_post_entreno['Fecha'] - t0_train).dt.total_seconds() / (3600 * 24)

            st.subheader("📅 Calendario de eventos reales vs simulados")
            generar_calendario_eventos(
                t0_train,
                events_real.values,
                events_sim,
                pd.to_datetime(fecha_sim_inicio),
                pd.to_datetime(fecha_sim_fin)
            )

            st.download_button(
                label="📥 Descargar imagen del calendario",
                data=open("calmap_feminicidios.png", "rb").read(),
                file_name="calmap_feminicidios.png",
                mime="image/png"
            )

            # NUEVO: mostrar lista de fechas simuladas
            st.subheader("📆 Fechas previstas de feminicidios (simulación)")
            df_fechas_sim = obtener_fechas_simuladas(t0_train, events_sim)
            st.dataframe(df_fechas_sim, use_container_width=True)

            csv_data = df_fechas_sim.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Descargar lista de fechas simuladas (CSV)",
                data=csv_data,
                file_name="fechas_simuladas.csv",
                mime="text/csv"
            )
