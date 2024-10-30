import pickle
import gradio as gr
import pandas as pd


df = pd.read_csv("./df.csv") 

# Cargar el modelo
with open('modelo.clf', 'rb') as file:
    modelo = pickle.load(file)

# Función de predicción
def predecir_resultado(equipo_local, equipo_visitante):
    try:
        if equipo_local == equipo_visitante:
            return "Error: El equipo local y visitante no pueden ser el mismo."
        
        columns_home = df.filter(like="ome", axis=1).columns
        columns_away = df.filter(like="way", axis=1).columns

        home_team = df.loc[df["homeTeam"] == equipo_local].loc[:, columns_home].iloc[-1, :]
        away_team = df.loc[df["awayTeam"] == equipo_visitante].loc[:, columns_away].iloc[-1, :]
        dataframe = pd.concat([home_team, away_team], axis=0)

        dataframe = pd.DataFrame(dataframe)

        dataframe = dataframe.T
        
        if dataframe.empty:
            return "No se encontraron datos para estos equipos."
        
        datos = modelo.feature_names_in_
        dataframe = dataframe[datos]

        prediccion = modelo.predict(dataframe)
        
        if prediccion == "DRAW":
            return "Empate"
        elif prediccion == "WINNER_HOME":
            return equipo_local
        elif prediccion == "WINNER_AWAY":
            return equipo_visitante
    except Exception as e:
        return f"Error: {e}"

equipos = df["homeTeam"].unique().tolist()

interfaz = gr.Interface(
    fn=predecir_resultado,
    inputs=[
        gr.Dropdown(label="Selecciona el equipo local", choices=equipos),
        gr.Dropdown(label="Selecciona el equipo visitante", choices=equipos)
    ],
    outputs=[gr.Textbox(label="Ganador")],
    title="Predicción del Resultado del Partido",
    description="Selecciona un equipo local y un equipo visitante para predecir el resultado."
)

interfaz.launch()