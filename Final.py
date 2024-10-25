import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import requests
from sklearn.dummy import DummyRegressor
from plotnine import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


#API Token
api = "d073f379a0e94969beb4956bda52ef0f"
headers = {"X-Auth-Token": api}

#Obtengo todos los datos
df = pd.DataFrame()
for year in range(2022, 2025):
    url = f'https://api.football-data.org/v4/competitions/PD/matches?season={year}'

    response = requests.get(url, headers=headers)
    data = response.json()

    matches = data["matches"]
    df_season = pd.DataFrame(matches)
    df = pd.concat([df, df_season], ignore_index=True)
    

#Formateando columna 'season'
df['startDate'] = df['season'].apply(lambda x: x['startDate'])
df['endDate'] = df['season'].apply(lambda x: x['endDate'])

df['startDate'] = pd.to_datetime(df['startDate'])
df['endDate'] = pd.to_datetime(df['endDate'])
df['season'] = df['startDate'].dt.year.astype(str) + '/' + df['endDate'].dt.year.astype(str).str[-2:]


#Formateando columnas 'awayTeam' y 'homeTeam'
df['homeTeam'] = df['homeTeam'].apply(lambda x: x['name'])
df['awayTeam'] = df['awayTeam'].apply(lambda x: x['name'])


#Formateando columna 'score' y añadiendo columnas de goles de local y visitante, resultado media parte y resultado final
df['homeGoalsFullTime'] = df['score'].apply(lambda x: x['fullTime']).apply(lambda x: x["home"])
df['awayGoalsFullTime'] = df['score'].apply(lambda x: x['fullTime']).apply(lambda x: x["away"])

df['homeGoalsHalfTime'] = df['score'].apply(lambda x: x['halfTime']).apply(lambda x: x["home"])
df['awayGoalsHalfTime'] = df['score'].apply(lambda x: x['halfTime']).apply(lambda x: x["away"])


#Formateando columna 'referees'
df["referees"] = df['referees'].apply(lambda x: x[0]['name'] if isinstance(x, list) and len(x) > 0 else None)

#Eliminando columnas irrelevantes
df = df.drop(["startDate", "endDate", "competition", "area", "odds", "id", "utcDate", "stage", "group", "lastUpdated", "score"], axis=1)

#Feature Engineering
teams = df["homeTeam"].unique()
seasons = df["season"].unique()

for team in teams:
    total_matches = df.loc[(df["awayTeam"] == team) | (df["homeTeam"] == team)]
    total_matches_finished = total_matches.loc[total_matches["status"] == "FINISHED"]
    total_matches_finished_len = len(total_matches_finished)
    total_away_team_goals = df.loc[(df["awayTeam"] == team)]["awayGoalsFullTime"].sum()
    total_home_team_goals = df.loc[(df["homeTeam"] == team)]["homeGoalsFullTime"].sum()
    total_goals = total_away_team_goals + total_home_team_goals
    df.loc[df["homeTeam"] == team, "totalHomeTeamGoals"] = total_goals
    df.loc[df["awayTeam"] == team, "totalAwayTeamGoals"] = total_goals
    df.loc[df["homeTeam"] == team, "homeTeamGoalRatio"] = total_goals / total_matches_finished_len
    df.loc[df["awayTeam"] == team, "awayTeamGoalRatio"] = total_goals / total_matches_finished_len
    df.loc[df["homeTeam"] == team, "totalHomeTeamGoalsAtHome"] = total_home_team_goals
    df.loc[df["awayTeam"] == team, "totalAwayTeamGoalsAway"] = total_away_team_goals
    df.loc[df["homeTeam"] == team, "homeTeamGoalRatioAtHome"] = total_home_team_goals / total_matches_finished_len
    df.loc[df["awayTeam"] == team, "awayTeamGoalRatioAway"] = total_away_team_goals / total_matches_finished_len
    total_home_team_goals_conceded = df.loc[df["homeTeam"] == team]["awayGoalsFullTime"].sum()
    total_away_team_goals_conceded = df.loc[df["awayTeam"] == team]["homeGoalsFullTime"].sum()
    total_goals_conceded = total_away_team_goals_conceded + total_home_team_goals_conceded
    df.loc[df["homeTeam"] == team, "totalHomeTeamGoalsConceded"] = total_goals_conceded
    df.loc[df["awayTeam"] == team, "totalAwayTeamGoalsConceded"] = total_goals_conceded
    df.loc[df["homeTeam"] == team, "homeTeamGoalConcededRatio"] = total_goals_conceded / total_matches_finished_len
    df.loc[df["awayTeam"] == team, "awayTeamGoalConcededRatio"] = total_goals_conceded / total_matches_finished_len
    df.loc[df["homeTeam"] == team, "totalHomeTeamGoalsConcededAtHome"] = total_home_team_goals_conceded
    df.loc[df["awayTeam"] == team, "totalAwayTeamGoalsConcededAway"] = total_away_team_goals_conceded
    df.loc[df["homeTeam"] == team, "homeTeamGoalConcededRatioAtHome"] = total_home_team_goals_conceded / total_matches_finished_len
    df.loc[df["awayTeam"] == team, "awayTeamGoalConcededRatioAway"] = total_away_team_goals_conceded / total_matches_finished_len

    for season in seasons:
        total_matches_finished_per_season = total_matches_finished.loc[total_matches_finished["season"] == season]
        total_matches_finished_per_season_len = len(total_matches_finished_per_season)
        total_home_team_goals_per_season = df.loc[(df["homeTeam"] == team) & (df["season"] == season)]["homeGoalsFullTime"].sum()
        total_away_team_goals_per_season = df.loc[(df["awayTeam"] == team) & (df["season"] == season)]["awayGoalsFullTime"].sum()
        total_home_team_goals_conceded_per_season = df.loc[(df["homeTeam"] == team) & (df["season"] == season)]["awayGoalsFullTime"].sum()
        total_away_team_goals_conceded_per_season = df.loc[(df["awayTeam"] == team) & (df["season"] == season)]["homeGoalsFullTime"].sum()
        total_goals_per_season = total_home_team_goals_per_season + total_away_team_goals_per_season
        total_goals_conceded_per_season = total_home_team_goals_conceded_per_season + total_away_team_goals_conceded_per_season
        df.loc[(df["homeTeam"] == team) & (df["season"] == season), "totalHomeTeamGoalsperSeason"] = total_goals_per_season
        df.loc[(df["awayTeam"] == team) & (df["season"] == season), "totalAwayTeamGoalsperSeason"] = total_goals_per_season
        df.loc[(df["homeTeam"] == team) & (df["season"] == season), "homeTeamGoalRatioperSeason"] = total_goals_per_season / total_matches_finished_per_season_len
        df.loc[(df["awayTeam"] == team) & (df["season"] == season), "awayTeamGoalRatioperSeason"] = total_goals_per_season / total_matches_finished_per_season_len
        df.loc[(df["homeTeam"] == team) & (df["season"] == season), "totalHomeTeamGoalsConcededperSeason"] = total_goals_conceded_per_season
        df.loc[(df["awayTeam"] == team) & (df["season"] == season), "totalAwayTeamGoalsConcededperSeason"] = total_goals_conceded_per_season
        df.loc[(df["homeTeam"] == team) & (df["season"] == season), "homeTeamGoalConcededRatioperSeason"] = total_goals_conceded_per_season / total_matches_finished_per_season_len
        df.loc[(df["awayTeam"] == team) & (df["season"] == season), "awayTeamGoalConcededRatioperSeason"] = total_goals_conceded_per_season / total_matches_finished_per_season_len



#Preprocesando

df_train = df.loc[df["status"] == "FINISHED"]
df_test = df.loc[df["status"] != "FINISHED"]

df_train.columns

columns_to_plot = ['homeTeamGoalRatio', 'awayTeamGoalRatio', 'homeTeamGoalConcededRatio', 'awayTeamGoalConcededRatio']
df_melted = df.melt(value_vars=columns_to_plot, var_name='GoalRatioType', value_name='Ratio')
ggplot(df_melted, aes(x='Ratio', fill='GoalRatioType')) + geom_histogram(alpha=0.6, bins=20, position='identity') + facet_wrap('~GoalRatioType', scales='free') + labs(title='Distribution of Goal Ratios: Scored and Conceded', x='Goal Ratio', y='Count') + theme_minimal()


rm_fcb_comparison = df.loc[(df["homeTeam"] == "Real Madrid CF") | (df["homeTeam"] == "FC Barcelona")]
ggplot(rm_fcb_comparison, aes(x='season', y='totalHomeTeamGoalsperSeason', color='homeTeam', group='homeTeam')) + geom_point() + geom_line() + labs(title='Total Goals Scored by Team per Season', x='Season', y='Total Goals') + theme_minimal()


#-----------------------------------
#MIGUEL
#-----------------------------------
#USAMOS DF_TRAIN (test son partidos que aun no han sucedido)

df_train2, df_test2 = train_test_split(df_train, test_size=0.30, random_state=42)


# Preserva las columnas originales antes de aplicar get_dummies
df_test_original = df_test2[["homeTeam", "awayTeam", "season"]].copy()


df_train2 = pd.get_dummies(df_train2,columns=["homeTeam","awayTeam","referees", "season"])
df_test2 = pd.get_dummies(df_test2,columns=["homeTeam","awayTeam","referees", "season"])

df_train2 = df_train2.drop(columns=["status"])
df_test2 = df_test2.drop(columns=["status"])


df_train2_copia = df_train2
df_test2_copia = df_test2

#GOLES LOCAL TRAIN
XL = df_train2.drop(columns=["homeGoalsFullTime"])
YL = df_train2.homeGoalsFullTime


#GOLES VISITANTE TRAIN
XV = df_train2_copia.drop(columns=["awayGoalsFullTime"])
YV = df_train2_copia.awayGoalsFullTime

#-------------------------
# ENTRENAMIENTO CON DecisionTreeRegressor
#-------------------------

# GOLES LOCAL
caja_negra_L_DT = DecisionTreeRegressor()
caja_negra_L_DT.fit(XL, YL)

# GOLES VISITANTE
caja_negra_V_DT = DecisionTreeRegressor()
caja_negra_V_DT.fit(XV, YV)

#-------------------------
# ENTRENAMIENTO CON RandomForestRegressor
#-------------------------

# GOLES LOCAL
caja_negra_L_RF = RandomForestRegressor(random_state=42)
caja_negra_L_RF.fit(XL, YL)

# GOLES VISITANTE
caja_negra_V_RF = RandomForestRegressor(random_state=42)
caja_negra_V_RF.fit(XV, YV)

#-------------------------------

# GOLES LOCAL TEST
XL_test = df_test2.drop(columns=["homeGoalsFullTime"])

# GOLES VISITANTE TEST
XV_test = df_test2_copia.drop(columns=["awayGoalsFullTime"])

#PREDICTS 
goles_local_DT = caja_negra_L_DT.predict(XL_test)
goles_visitante_DT = caja_negra_V_DT.predict(XV_test)

goles_local_RF = caja_negra_L_RF.predict(XL_test)
goles_visitante_RF = caja_negra_V_RF.predict(XV_test)


#-------------------------
# EVALUACIÓN
#-------------------------

# Calcula métricas para Decision Tree
mse_local_DT = mean_squared_error(df_test2["homeGoalsFullTime"], goles_local_DT)
mse_visitante_DT = mean_squared_error(df_test2["awayGoalsFullTime"], goles_visitante_DT)
mae_local_DT = mean_absolute_error(df_test2["homeGoalsFullTime"], goles_local_DT)
mae_visitante_DT = mean_absolute_error(df_test2["awayGoalsFullTime"], goles_visitante_DT)
r2_local_DT = r2_score(df_test2["homeGoalsFullTime"], goles_local_DT)
r2_visitante_DT = r2_score(df_test2["awayGoalsFullTime"], goles_visitante_DT)

# Calcula métricas para Random Forest
mse_local_RF = mean_squared_error(df_test2["homeGoalsFullTime"], goles_local_RF)
mse_visitante_RF = mean_squared_error(df_test2["awayGoalsFullTime"], goles_visitante_RF)
mae_local_RF = mean_absolute_error(df_test2["homeGoalsFullTime"], goles_local_RF)
mae_visitante_RF = mean_absolute_error(df_test2["awayGoalsFullTime"], goles_visitante_RF)
r2_local_RF = r2_score(df_test2["homeGoalsFullTime"], goles_local_RF)
r2_visitante_RF = r2_score(df_test2["awayGoalsFullTime"], goles_visitante_RF)


#MSE Y MAE cuanto menores mejor
#R2 valores cercanos a 1

print("Resultados para Decision Tree:")
print(f"MSE Local: {mse_local_DT:.2f}, Visitante: {mse_visitante_DT:.2f}")
print(f"MAE Local: {mae_local_DT:.2f}, Visitante: {mae_visitante_DT:.2f}")
print(f"R2 Local: {r2_local_DT:.2f}, Visitante: {r2_visitante_DT:.2f}\n")

print("Resultados para Random Forest:")
print(f"MSE Local: {mse_local_RF:.2f}, Visitante: {mse_visitante_RF:.2f}")
print(f"MAE Local: {mae_local_RF:.2f}, Visitante: {mae_visitante_RF:.2f}")
print(f"R2 Local: {r2_local_RF:.2f}, Visitante: {r2_visitante_RF:.2f}")


#Paso los valores de Random Forest (mejores) al DF resultado
resultado = pd.DataFrame({
    "homeTeam": df_test_original["homeTeam"].astype(str),
    "awayTeam": df_test_original["awayTeam"].astype(str),
    "season": df_test_original["season"].astype(str),
    "predicted_homeGoals": goles_local_RF,
    "predicted_awayGoals": goles_visitante_RF
})
