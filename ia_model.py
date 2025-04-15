import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

class IAGradientBoostingPredictor:
    def __init__(self, n_periodos=20, taxa_de_analise=1):
        self.n_periodos = n_periodos
        self.taxa_de_analise = taxa_de_analise

    def process_and_predict(self, json_data):
        df = pd.DataFrame(json_data)

        # Detectar coluna de tempo
        df.index = pd.to_datetime(df.iloc[:, 0]) if not isinstance(df.index, pd.DatetimeIndex) else df.index
        df = df.drop(df.columns[0], axis=1)  # remove coluna de tempo se já for índice

        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='last')]

        value_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        df = df[value_cols].dropna()

        # Features derivadas
        for col in value_cols:
            df[f"{col}_diff1"] = df[col].diff()
            df[f"{col}_ma3"] = df[col].rolling(3).mean()
            df[f"{col}_ma5"] = df[col].rolling(5).mean()
        df.dropna(inplace=True)
        feature_cols = df.columns.tolist()

        # Parâmetros
        total_periods = len(df)
        window_size = max(24, min(int(total_periods * self.taxa_de_analise), 1000))
        granularidade = df.index.to_series().diff().median()

        # Normalização
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        def create_dataset(data, window_size):
            X, y = [], []
            for i in range(len(data) - window_size):
                X.append(data[i:i+window_size].flatten())
                y.append(data[i+window_size][:len(value_cols)])
            return np.array(X), np.array(y)

        X, y = create_dataset(scaled_data, window_size)
        X_train, y_train = X[:int(len(X)*0.8)], y[:int(len(X)*0.8)]

        model = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=300,
                max_depth=7,
                learning_rate=0.05,
                random_state=42
            )
        )
        model.fit(X_train, y_train)

        # Previsão histórica
        previsoes_historico_scaled = model.predict(X)
        previsoes_historico = scaler.inverse_transform(
            np.hstack([
                previsoes_historico_scaled,
                np.zeros((len(previsoes_historico_scaled), len(feature_cols) - len(value_cols)))
            ])
        )[:, :len(value_cols)]

        mse_historico = {}
        for i, col in enumerate(value_cols):
            real = df[col].values[window_size:]
            pred = previsoes_historico[:, i]
            mse_historico[col] = mean_squared_error(real, pred)

        # Projeção futura
        janela_atual = scaled_data[-window_size:].copy()
        futuro_escalado = []

        for _ in range(self.n_periodos):
            janela_input = janela_atual.flatten().reshape(1, -1)
            pred = model.predict(janela_input)[0]
            futuro_escalado.append(pred)

            linha_futura = np.zeros(scaled_data.shape[1])
            linha_futura[:len(value_cols)] = pred
            janela_atual = np.vstack([janela_atual[1:], linha_futura])

        futuro = scaler.inverse_transform(
            np.hstack([futuro_escalado, np.zeros((self.n_periodos, len(feature_cols) - len(value_cols)))])
        )[:, :len(value_cols)]

        datas_futuras = [df.index[-1] + granularidade * (i + 1) for i in range(self.n_periodos)]

        resultado_df = pd.DataFrame(futuro, columns=value_cols, index=datas_futuras)

        return resultado_df, mse_historico
