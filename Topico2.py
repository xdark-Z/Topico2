import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.interpolate import griddata 
import time
import math
import warnings

# --- CORRECCIÓN DE ERROR PANDAS: SettingWithCopyWarning ---
# Se utiliza un try/except para manejar la ubicación de la advertencia 
# que cambió de 'pandas.core.common' a 'pandas.errors' en Pandas >= 1.5.
try:
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
except AttributeError:
    # Fallback para versiones más antiguas, aunque el error indica el problema con la versión antigua.
    pass


# --- Librerías para Machine Learning ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin

## <--- CAMBIOS PYTORCH/CUDA ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
## --- FIN CAMBIOS PYTORCH/CUDA ---

# Intento de detectar PyTorch/CUDA (Informativo)
DEVICE = 'cpu'
try:
    if torch.cuda.is_available():
        DEVICE = 'cuda'
except ImportError:
    pass

# ----------------------------------------
# --- 0. IMPLEMENTACIÓN DE MODELOS PYTORCH ---
# ----------------------------------------

# Diccionario de Activaciones
ACTIVATIONS = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'LeakyReLU': nn.LeakyReLU,
    'ELU': nn.ELU
}

# Diccionario de Inicializadores
INITIALIZERS = {
    'Xavier Uniform': nn.init.xavier_uniform_,
    'Kaiming Uniform': nn.init.kaiming_uniform_,
    'Uniform': nn.init.uniform_,
    'Normal': nn.init.normal_
}

# 0.1 Red Neuronal Profunda Determinista (DNN)
class DeterministicNeuralNet(nn.Module):
    """
    Red Neuronal Profunda (DNN) - Modelo Determinista.
    """
    def __init__(self, input_size, hidden_layers, output_size=1, dropout_rate=0.1, activation_func='ReLU', initializer_method='Xavier Uniform'):
        super(DeterministicNeuralNet, self).__init__()
        
        layers = []
        current_size = input_size
        activation = ACTIVATIONS.get(activation_func, nn.ReLU)()
        initializer = INITIALIZERS.get(initializer_method, nn.init.xavier_uniform_)
        
        # Crear capas ocultas dinámicamente
        for hidden_size in hidden_layers:
            linear_layer = nn.Linear(current_size, hidden_size)
            
            # Aplicar inicialización
            initializer(linear_layer.weight)
            nn.init.zeros_(linear_layer.bias)
            
            layers.append(linear_layer)
            layers.append(activation) # Activación seleccionada
            layers.append(nn.Dropout(dropout_rate)) # Dropout para regularización
            current_size = hidden_size
        
        # Capa de salida
        final_layer = nn.Linear(current_size, output_size)
        initializer(final_layer.weight) # Inicialización de la capa final
        nn.init.zeros_(final_layer.bias) 
        layers.append(final_layer)
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# 0.2 Red Neuronal Bayesiana (BNN) - Implementación simplificada con Dropout Monte Carlo (MC-Dropout)
class BayesianNeuralNet(nn.Module):
    """
    Red Neuronal Bayesiana (BNN) - Utiliza MC-Dropout para cuantificación de incertidumbre.
    """
    def __init__(self, input_size, hidden_layers, output_size=1, dropout_rate=0.1, activation_func='ReLU', initializer_method='Xavier Uniform'):
        super(BayesianNeuralNet, self).__init__()
        
        layers = []
        current_size = input_size
        activation = ACTIVATIONS.get(activation_func, nn.ReLU)()
        initializer = INITIALIZERS.get(initializer_method, nn.init.xavier_uniform_)
        
        # Crear capas ocultas dinámicamente (Con dropout que se mantiene activo en test)
        for hidden_size in hidden_layers:
            linear_layer = nn.Linear(current_size, hidden_size)
            initializer(linear_layer.weight)
            nn.init.zeros_(linear_layer.bias)
            
            layers.append(linear_layer)
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate)) # Dropout activo
            current_size = hidden_size
        
        # Capa de salida
        final_layer = nn.Linear(current_size, output_size)
        initializer(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        # Mantenemos el dropout activo durante la inferencia para MC-Dropout
        return self.model(x)

# 0.3 Wrapper para integrarse con Scikit-learn Pipeline - Actualizado
class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layers=[64, 32], epochs=100, batch_size=32, learning_rate=0.001, device='cpu',
                 dropout_rate=0.1, activation_func='ReLU', optimizer_choice='Adam', weight_decay=0.0,
                 momentum=0.9, initializer_method='Xavier Uniform', loss_fn='MSELoss', scheduler_choice='Ninguno',
                 model_type='DNN (Determinista)', num_monte_carlo_samples=100):
        
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.dropout_rate = dropout_rate
        self.activation_func = activation_func
        self.optimizer_choice = optimizer_choice
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.initializer_method = initializer_method
        self.loss_fn = loss_fn
        self.scheduler_choice = scheduler_choice
        self.model_type = model_type 
        self.num_monte_carlo_samples = num_monte_carlo_samples 
        
        self.input_size_ = None
        self.model = None

    def fit(self, X, y):
        self.input_size_ = X.shape[1]
        
        # Selección de Arquitectura de Red
        if self.model_type == 'BNN (MC-Dropout)':
            NetClass = BayesianNeuralNet
        else: # DNN por defecto
            NetClass = DeterministicNeuralNet

        self.model = NetClass(
            self.input_size_, 
            self.hidden_layers, 
            dropout_rate=self.dropout_rate, 
            activation_func=self.activation_func,
            initializer_method=self.initializer_method
        ).to(self.device)
        
        # Criterio de Pérdida (Loss)
        if self.loss_fn == 'MAELoss':
            criterion = nn.L1Loss() # MAE
        else:
            criterion = nn.MSELoss() # MSE Default
            
        # Optimizador
        if self.optimizer_choice == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_choice == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        elif self.optimizer_choice == 'RMSprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            
        # Scheduler
        if self.scheduler_choice == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(self.epochs * 0.3)), gamma=0.1)
        else:
            scheduler = None


        # Preparar datos
        X_tensor = torch.tensor(X.astype(np.float32)).to(self.device)
        y_tensor = torch.tensor(y.values.astype(np.float32)).unsqueeze(1).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            if scheduler:
                scheduler.step()
        
        return self

    def predict(self, X, return_uncertainty=False, confidence_level=0.95):
        self.model.eval()
        
        X_tensor = torch.tensor(X.astype(np.float32)).to(self.device)
        
        # Si es DNN o si no se requiere incertidumbre, predicción determinista
        if self.model_type == 'DNN (Determinista)' or not return_uncertainty:
            with torch.no_grad():
                predictions = self.model(X_tensor).cpu().numpy().flatten()
            return predictions
        
        # Si es BNN y se requiere incertidumbre (MC-Dropout)
        elif self.model_type == 'BNN (MC-Dropout)' and return_uncertainty:
            
            # Forzar el modo 'train' para mantener el dropout activo durante la inferencia
            self.model.train() 
            
            all_predictions = []
            with torch.no_grad():
                for _ in range(self.num_monte_carlo_samples):
                    output = self.model(X_tensor).cpu().numpy().flatten()
                    all_predictions.append(output)
            
            all_predictions = np.array(all_predictions) # Forma (num_samples, num_data)
            
            # Calcular la media
            mean_predictions = np.mean(all_predictions, axis=0)
            
            # Cuantiles para el intervalo de confianza (ej: 95% -> 2.5% y 97.5%)
            alpha = (1 - confidence_level) / 2
            lower_bound = np.percentile(all_predictions, alpha * 100, axis=0)
            upper_bound = np.percentile(all_predictions, (1 - alpha) * 100, axis=0)
            
            # Devolvemos un DataFrame con los tres valores
            results_df = pd.DataFrame({
                'Y_Predicho_Media': mean_predictions,
                'Y_Predicho_Inferior': lower_bound,
                'Y_Predicho_Superior': upper_bound
            })
            
            return results_df


## --- FIN IMPLEMENTACIÓN DE MODELOS PYTORCH ---


# ----------------------------------------
# --- FUNCIONES GLOBALES DE PREDICCIÓN (ACTUALIZADAS) ---
# ----------------------------------------

def make_prediction(new_X_data, target_name, model_type, confidence_level=0.95):
    """Aplica el preprocesador y el modelo entrenado a los nuevos datos X."""
    
    if st.session_state.trained_regressor is None or st.session_state.trained_preprocessor is None:
        st.error("Error: Modelo o Preprocesador no disponibles. Por favor, entrene el modelo primero.")
        return None, None
    
    # Preprocesamiento de datos nuevos (relleno y type casting)
    # Convertir columnas categóricas a tipo 'object' para que el preprocesador no falle
    for col in st.session_state.categorical_features_ml:
        if col in new_X_data.columns:
            new_X_data[col] = new_X_data[col].fillna('missing_value').astype(object)
        else:
            new_X_data[col] = 'missing_value' 
            
    # Asegurarse de que las columnas numéricas mantengan su tipo 
    for col in st.session_state.numeric_features_ml:
        if col not in new_X_data.columns:
            new_X_data[col] = np.nan
        else:
            new_X_data[col] = pd.to_numeric(new_X_data[col], errors='coerce')


    # Asegurar el orden de las columnas 
    new_X_data = new_X_data[st.session_state.exogenous_cols]
    
    try:
        # 1. Aplicar preprocesamiento (SOLO transform)
        X_sim_processed = st.session_state.trained_preprocessor.transform(new_X_data)
        
        # 2. Predecir
        if model_type == 'BNN (MC-Dropout)':
            # Predicción BNN devuelve un DataFrame
            Y_pred_results = st.session_state.trained_regressor.predict(
                X_sim_processed, 
                return_uncertainty=True, 
                confidence_level=confidence_level
            )
            Y_pred = Y_pred_results['Y_Predicho_Media'].values
            
            # 3. Crear DataFrame de resultados para BNN
            prediction_df = new_X_data.copy()
            prediction_df.insert(0, f"Predicción ({target_name})", Y_pred)
            prediction_df.insert(1, f"Límite Inferior ({confidence_level*100:.0f}%)", Y_pred_results['Y_Predicho_Inferior'].values)
            prediction_df.insert(2, f"Límite Superior ({confidence_level*100:.0f}%)", Y_pred_results['Y_Predicho_Superior'].values)
            
        else:
            # Predicción DNN devuelve un array
            Y_pred = st.session_state.trained_regressor.predict(X_sim_processed, return_uncertainty=False)
            
            # 3. Crear DataFrame de resultados para DNN
            prediction_df = new_X_data.copy()
            prediction_df.insert(0, f"Predicción ({target_name})", Y_pred)

        return Y_pred, prediction_df
    except Exception as e:
        st.error(f"Error durante la predicción/preprocesamiento. Asegúrese de que las columnas coincidan con las variables predictoras utilizadas en el entrenamiento.")
        st.exception(e) 
        return None, None

def plot_prediction_graph(prediction_df, target_name, y_type, model_type, confidence_level=0.95):
    """Genera un gráfico de línea para la predicción de nuevos escenarios, incluyendo IC para BNN."""
    if prediction_df is None:
        return

    fig = go.Figure()

    if model_type == 'BNN (MC-Dropout)':
        lower_col = f"Límite Inferior ({confidence_level*100:.0f}%)"
        upper_col = f"Límite Superior ({confidence_level*100:.0f}%)"
        pred_col = f"Predicción ({target_name})"
        
        # Trazar el área de incertidumbre (Intervalo de Confianza)
        fig.add_trace(go.Scatter(
            x=prediction_df.index.tolist() + prediction_df.index.tolist()[::-1],
            y=prediction_df[upper_col].tolist() + prediction_df[lower_col].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(150, 50, 200, 0.2)', # Púrpura claro
            line=dict(color='rgba(255,255,255,0)'),
            name=f'IC {confidence_level*100:.0f}%'
        ))

        # Trazar la predicción media
        fig.add_trace(go.Scatter(
            x=prediction_df.index, 
            y=prediction_df[pred_col], 
            mode='lines+markers', 
            name=f"Predicción Media ({target_name})", 
            line=dict(color='purple', width=3)
        ))

    else:
        # Gráfico estándar para DNN
        fig.add_trace(go.Scatter(
            x=prediction_df.index, 
            y=prediction_df[f"Predicción ({target_name})"], 
            mode='lines+markers', 
            name=f"Predicción de {target_name}", 
            line=dict(color='purple', width=3)
        ))

    fig.update_layout(
        title=f"Predicción de Avance para Nuevos Escenarios ({model_type})",
        xaxis_title="N° Dato / Escenario",
        yaxis_title=f"Valor Predicho ({target_name})",
        height=500,
        yaxis=dict(type=y_type)
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------
# --- CONFIGURACIÓN DE LA PÁGINA (NUEVA) ---
# ----------------------------------------
st.set_page_config(page_title="IA para la Estimación de Avances en Túneles Mineros", layout="wide")

st.title("IA para la Estimación de Avances en Túneles Mineros")
st.markdown("Plataforma para el análisis de datos de operación minera y modelado predictivo del avance.")

# Inicializar variables de sesión (sin cambios lógicos)
if 'data' not in st.session_state: st.session_state.data = None
if 'filtered_data' not in st.session_state: st.session_state.filtered_data = None
if 'df_current' not in st.session_state: st.session_state.df_current = None
if 'filter_column' not in st.session_state: st.session_state.filter_column = None
if 'selected_categories' not in st.session_state: st.session_state.selected_categories = []
if 'use_binning_global' not in st.session_state: st.session_state.use_binning_global = False
if 'bin_width_global' not in st.session_state: st.session_state.bin_width_global = 50.0
if 'agg_method_global' not in st.session_state: st.session_state.agg_method_global = "Promedio"
if 'bin_base_col' not in st.session_state: st.session_state.bin_base_col = None
# Nuevas variables de sesión para el modelo y el preprocesador
if 'trained_regressor' not in st.session_state: st.session_state.trained_regressor = None
if 'trained_preprocessor' not in st.session_state: st.session_state.trained_preprocessor = None
if 'target_name' not in st.session_state: st.session_state.target_name = None
if 'exogenous_cols' not in st.session_state: st.session_state.exogenous_cols = None
if 'numeric_features_ml' not in st.session_state: st.session_state.numeric_features_ml = None
if 'categorical_features_ml' not in st.session_state: st.session_state.categorical_features_ml = None
if 'feature_importance_df' not in st.session_state: st.session_state.feature_importance_df = None
# Variables de sesión para el tipo de modelo entrenado
if 'trained_model_type' not in st.session_state: st.session_state.trained_model_type = 'DNN (Determinista)'
if 'trained_confidence_level' not in st.session_state: st.session_state.trained_confidence_level = 0.95

# Paleta de colores para categorías
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# --- FUNCIÓN DE AGRUPACIÓN (Mismo código que el original) ---
def apply_aggregation(df_input, bin_width, agg_method, x_bin_col):
    """Aplica agregación a TODAS las columnas numéricas usando x_bin_col como base para el intervalo (ej: Longitud de Túnel)."""
    if df_input is None or df_input.empty or x_bin_col is None:
        return df_input, []

    numeric_cols_base = df_input.select_dtypes(include=[np.number]).columns.tolist()
    
    if x_bin_col not in numeric_cols_base:
        st.error(f"La columna base '{x_bin_col}' no es numérica.")
        return df_input, []

    df_agg = df_input.copy()
    
    min_val = df_agg[x_bin_col].min()
    max_val = df_agg[x_bin_col].max()
    
    if bin_width <= 0 or max_val <= min_val:
        st.error("Configuración de Agregación no válida o datos insuficientes.")
        return df_input, []
        
    # Crear bins
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    
    # Crear etiquetas: Usamos el punto medio del bin para la nueva "columna de índice"
    bin_labels = [f"{(bins[i] + bins[i+1])/2:.2f}" for i in range(len(bins)-1)]
    
    # Aplicar binning a la columna base
    df_agg['X_Agg_Label'] = pd.cut(
        df_agg[x_bin_col], 
        bins=bins, 
        include_lowest=True, 
        labels=bin_labels if len(bin_labels) == len(bins)-1 else False, # Usar etiquetas solo si el número coincide
        right=True
    )
    
    if df_agg['X_Agg_Label'].isnull().all():
        st.warning("Advertencia: La Agregación por Intervalos no pudo crear etiquetas válidas. Revise el Ancho del Intervalo.")
        return df_input, []
    
    # Mapeo de agregación
    agg_map = {
        "Suma": 'sum',
        "Promedio": 'mean',
        "Conteo": 'count'
    }
    agg_func = agg_map.get(agg_method, 'mean')
    
    # 2. Agregación: Aplicar la función de agregación a TODAS las columnas numéricas
    cols_to_agg = [c for c in numeric_cols_base]
    
    # Crea un diccionario de agregación: {columna: función}
    agg_dict = {col: agg_func for col in cols_to_agg}
    
    # Agregación por la etiqueta del Bin
    df_aggregated = df_agg.groupby('X_Agg_Label', observed=True).agg(agg_dict).reset_index()
    
    # Renombrar la columna de la etiqueta de bin
    new_bin_col_name = f"{x_bin_col}_Centro_Intervalo"
    df_aggregated.rename(columns={'X_Agg_Label': new_bin_col_name}, inplace=True)
    
    st.success(f"Datos Agrupados: {len(df_aggregated)} filas (Método: **{agg_method}**). Base: **{x_bin_col}**.")
    
    # Devuelve el nuevo DF y la columna que representa el bin
    return df_aggregated, [new_bin_col_name] + [c for c in df_aggregated.columns if c != new_bin_col_name]

# --- FIN FUNCIÓN DE AGRUPACIÓN ---

# Sidebar para carga de datos y configuración
with st.sidebar:
    st.header("1. Carga de Datos de Operación")
    uploaded_file = st.file_uploader("Seleccione archivo Excel", type=['xlsx', 'xls'])
    
    if uploaded_file:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        selected_sheet = st.selectbox("Seleccione la hoja de datos", sheet_names)
        
        if st.button("Cargar Datos"):
            # Resetear el modelo entrenado al cargar nuevos datos
            st.session_state.trained_regressor = None
            st.session_state.data = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            st.session_state.filtered_data = st.session_state.data.copy()
            st.session_state.df_current = st.session_state.data.copy()
            st.session_state.filter_column = None
            st.session_state.selected_categories = []
            
            numeric_cols_init = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols_init:
                st.session_state.bin_base_col = numeric_cols_init[0]
            else:
                st.session_state.bin_base_col = None

            st.success(f"Datos cargados: {len(st.session_state.data)} filas")

    # --- NUEVA FUNCIONALIDAD: POBLAMIENTO DE DATOS ESTADÍSTICO ---
    if st.session_state.data is not None:
        st.divider()
        st.header("1.1. Poblamiento de Datos (Estadístico)")
        
        with st.expander("🛠️ Configurar Generación de Datos"):
            st.info("Esta opción permite aumentar la cantidad de datos (Data Augmentation) basándose estrictamente en las propiedades estadísticas (Media, Varianza, Covarianza) de sus variables numéricas actuales.")
            
            # Inputs para la generación
            n_samples_gen = st.number_input("Cantidad de datos a generar (Filas)", min_value=10, value=100, step=10, help="Número de nuevas filas sintéticas a crear.")
            
            method_gen = st.selectbox(
                "Método de Generación Estadística", 
                [
                    "Multivariante (Conserva Correlaciones)", 
                    "Bootstrap con Ruido (Variación de existentes)"
                ],
                help="Multivariante: Crea datos nuevos usando la matriz de covarianza (ideal para mantener relaciones matemáticas). Bootstrap: Remuestrea datos existentes y añade ruido leve."
            )
            
            noise_level = 0.01
            if "Bootstrap" in method_gen:
                noise_level = st.slider("Nivel de Ruido (Perturbación)", 0.0, 0.2, 0.01, format="%.3f", help="Porcentaje de variación aleatoria añadida a los datos clonados.")
            
            if st.button("Generar y Anexar Datos", use_container_width=True):
                df_original = st.session_state.data.copy()
                
                # Identificar columnas numéricas
                numeric_cols_gen = df_original.select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_cols_gen:
                    st.error("No se encontraron columnas numéricas para generar datos.")
                else:
                    try:
                        new_data_dict = {}
                        
                        # 1. Lógica para Variables Numéricas
                        df_num = df_original[numeric_cols_gen].dropna()
                        
                        if method_gen == "Multivariante (Conserva Correlaciones)":
                            # Calculamos Media y Covarianza
                            means = df_num.mean()
                            cov_matrix = df_num.cov()
                            
                            # Generación usando distribución normal multivariante
                            # Esto preserva la correlación entre variables (ej: si X sube, Y baja)
                            generated_data = np.random.multivariate_normal(means, cov_matrix, n_samples_gen)
                            generated_df_num = pd.DataFrame(generated_data, columns=numeric_cols_gen)
                            
                        else: # Bootstrap con Ruido
                            # Muestreo con reemplazo
                            generated_df_num = df_num.sample(n=n_samples_gen, replace=True).reset_index(drop=True)
                            
                            # Añadir ruido Gaussiano
                            noise = np.random.normal(0, noise_level, generated_df_num.shape)
                            # Multiplicamos por la desviación estándar para que el ruido sea proporcional a la escala de cada variable
                            scale_factors = generated_df_num.std()
                            generated_df_num = generated_df_num + (noise * scale_factors.values)

                        # 2. Manejo de Columnas NO Numéricas (si las hubiera, aunque el usuario dijo que no)
                        # Simplemente hacemos un muestreo aleatorio de las categorías existentes para rellenar
                        non_numeric_cols = [c for c in df_original.columns if c not in numeric_cols_gen]
                        if non_numeric_cols:
                            df_cat = df_original[non_numeric_cols]
                            generated_df_cat = df_cat.sample(n=n_samples_gen, replace=True).reset_index(drop=True)
                            df_synthetic = pd.concat([generated_df_num, generated_df_cat], axis=1)
                        else:
                            df_synthetic = generated_df_num

                        # Asegurar orden de columnas
                        df_synthetic = df_synthetic[df_original.columns]
                        
                        # Anexar a los datos originales en st.session_state
                        st.session_state.data = pd.concat([st.session_state.data, df_synthetic], ignore_index=True)
                        
                        # Actualizar dataframes derivados para que el resto de la app vea los cambios
                        st.session_state.filtered_data = st.session_state.data.copy()
                        st.session_state.df_current = st.session_state.data.copy()
                        
                        # Resetear filtros y modelos ya que los datos cambiaron
                        st.session_state.trained_regressor = None 
                        st.session_state.filter_column = None
                        st.session_state.selected_categories = []
                        
                        st.success(f"✅ ¡Éxito! Se generaron {n_samples_gen} registros sintéticos. Total datos: {len(st.session_state.data)}.")
                        time.sleep(1) # Breve pausa para leer el mensaje
                        st.rerun() # Recargar la app para refrescar gráficos
                        
                    except Exception as e:
                        st.error(f"Error en la generación estadística: {str(e)}")
                        st.warning("Intente usar el método Bootstrap si la matriz de covarianza es singular.")
    
    # --- FIN NUEVA FUNCIONALIDAD ---

    st.divider()
    
    # Detección de Aceleración (PyTorch/CUDA)
    if DEVICE == 'cuda':
        st.success(f"Aceleración GPU CUDA detectada: {torch.cuda.get_device_name(0)}")
    elif DEVICE == 'cpu':
        st.info("Aceleración de PyTorch en CPU (Sin CUDA)")
    
    st.divider()
    
    # Configuración de Filtros
    if st.session_state.data is not None:
        st.header("2. Opciones de Filtro y Agregación")
        
        # Filtros por categoría
        st.subheader("Filtro Categórico")
        use_filter = st.checkbox("Aplicar filtro por variable categórica")
        
        if use_filter:
            categorical_cols = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
            
            if categorical_cols:
                filter_column = st.selectbox("Variable Categórica", categorical_cols)
                st.session_state.filter_column = filter_column
                unique_values = st.session_state.data[filter_column].unique().tolist()
                selected_values = st.multiselect("Seleccionar Categorías", unique_values, default=unique_values)
                st.session_state.selected_categories = selected_values
                
                if selected_values:
                    st.session_state.filtered_data = st.session_state.data[
                        st.session_state.data[filter_column].isin(selected_values)
                    ]
                    st.info(f"Datos filtrados: {len(st.session_state.filtered_data)} filas")
                else:
                    st.session_state.filtered_data = st.session_state.data.copy() 
            else:
                st.warning("No hay columnas categóricas disponibles para filtrar.")
        else:
            st.session_state.filtered_data = st.session_state.data.copy()
            st.session_state.filter_column = None
            st.session_state.selected_categories = []
        
        st.markdown("---")
        
        # Agregación / Binning (Global)
        st.subheader("Agregación por Intervalos")
        st.session_state.use_binning_global = st.checkbox(
            "Activar Agregación Global (para series de tiempo/distancia)", 
            key="use_binning_global_checkbox_new"
        )
        
        current_numeric_cols = st.session_state.filtered_data.select_dtypes(include=[np.number]).columns.tolist()

        if st.session_state.use_binning_global:
            if not current_numeric_cols:
                st.warning("No hay columnas numéricas para aplicar la agregación.")
                st.session_state.use_binning_global = False
            else:
                st.session_state.bin_base_col = st.selectbox(
                    "Columna Base para Intervalos (Eje X)",
                    current_numeric_cols,
                    index=current_numeric_cols.index(st.session_state.bin_base_col) if st.session_state.bin_base_col in current_numeric_cols else 0,
                    key="bin_base_col_select_new"
                )
                
                st.session_state.bin_width_global = st.number_input(
                    "Ancho del Intervalo (Unidad de X)", 
                    min_value=0.01, 
                    value=st.session_state.bin_width_global, 
                    step=1.0, 
                    key="bin_width_global_input_new",
                    help=f"Define el ancho de cada intervalo de agregación para la variable '{st.session_state.bin_base_col}'."
                )
                st.session_state.agg_method_global = st.selectbox(
                    "Método de Agregación", 
                    ["Suma", "Promedio", "Conteo"],
                    key="agg_method_global_select_new"
                )
                
                # Aplicar Agregación
                df_aggregated, new_cols = apply_aggregation(
                    st.session_state.filtered_data, 
                    st.session_state.bin_width_global, 
                    st.session_state.agg_method_global,
                    st.session_state.bin_base_col
                )
                st.session_state.df_current = df_aggregated
                
                if st.session_state.filter_column:
                    st.info("El filtro categórico se anula al usar Agregación Global.")
        else:
            st.session_state.df_current = st.session_state.filtered_data.copy()

# Verificar si hay datos cargados
if st.session_state.data is None:
    st.info("Por favor, cargue un archivo de datos para comenzar el análisis.")
    st.stop()

# --- DATAFRAME DE TRABAJO ---
df = st.session_state.df_current
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df.columns.tolist()

# Excluir columnas de fecha/hora de ML
unsupported_dtypes = ['datetime64', 'datetime64[ns]']
valid_cols_for_ml = [col for col in df.columns if df[col].dtype not in unsupported_dtypes]


# ----------------------------------------
# --- PESTAÑAS PRINCIPALES (AJUSTADAS) ---
# ----------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Visualización 2D (Series)", 
    "Visualización 3D", 
    "Análisis Descriptivo (EDA)", 
    "Análisis de Correlación", 
    "Modelo Predictivo (DNN/BNN)"
])

# --- TAB 1: Visualización 2D ---
with tab1:
    st.header("Análisis de Series de Avance (2D)")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Configuración de Ejes y Datos")
        plot_type = st.radio("Tipo de Gráfico", ["Línea", "Puntos"])
        
        # Eje X
        x_axis = st.selectbox("Eje X (Base de la Serie)", all_cols, key="x_axis_new")
        x_axis_scale = st.radio("Escala Eje X", ["Lineal", "Logarítmica"], key="x_axis_scale_new")
        
        # Eje Y Primario
        st.write("**Eje Y Primario**")
        y_primary = st.multiselect("Variables a Trazar (Eje Izquierdo)", numeric_cols, key="y_primary_new")
        y_primary_scale = st.radio("Escala Y Primario", ["Lineal", "Logarítmica"], key="y_primary_scale_new")
        
        # Eje Y Secundario
        use_secondary = st.checkbox("Usar Eje Y Secundario (Derecha)")
        y_secondary = []
        y_secondary_scale = "Lineal"
        
        if use_secondary:
            y_secondary = st.multiselect("Variables a Trazar (Eje Derecho)", numeric_cols, key="y_secondary_new")
            y_secondary_scale = st.radio("Escala Y Secundario", ["Lineal", "Logarítmica"], key="y_secondary_scale_new")
        
        # Media móvil (Solo si NO hay agregación global)
        st.divider()
        if st.session_state.use_binning_global:
            current_use_ma = False
            st.warning("La Media Móvil se deshabilita al usar Agregación por Intervalos.")
        else:
            current_use_ma = st.checkbox("Aplicar Media Móvil (Suavizado)")
            window_size = 5
            if current_use_ma:
                window_size = st.slider("Ventana de Media Móvil", 2, 50, 5)

    with col2:
        if y_primary or y_secondary:
            
            df_plot = df
            
            if use_secondary and y_secondary:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
            else:
                fig = go.Figure()
            
            def add_traces(dataframe, x_col, y_cols, is_secondary, category=None, color=None):
                traces = []
                for y_col in y_cols:
                    y_data = dataframe[y_col]
                    
                    if current_use_ma and y_col in numeric_cols:
                        y_plot = y_data.rolling(window=window_size, min_periods=1).mean()
                        name = f"{y_col}" + (f" - {category}" if category else "") + f" (MA{window_size})"
                    else:
                        y_plot = y_data
                        name = f"{y_col}" + (f" - {category}" if category else "")
                    
                    mode = 'lines' if plot_type == "Línea" else 'markers'
                    
                    line_style = dict(dash='dash', color=color) if is_secondary and plot_type == "Línea" else dict(color=color)
                    marker_style = dict(symbol='diamond', color=color) if is_secondary and plot_type == "Puntos" else dict(color=color)
                    
                    trace = go.Scatter(
                        x=dataframe[x_col], 
                        y=y_plot, 
                        mode=mode, 
                        name=name, 
                        line=line_style,
                        marker=marker_style
                    )
                    traces.append(trace)
                return traces

            if st.session_state.filter_column and st.session_state.selected_categories and not st.session_state.use_binning_global:
                for cat_idx, category in enumerate(st.session_state.selected_categories):
                    df_cat = df_plot[df_plot[st.session_state.filter_column] == category]
                    color = COLORS[cat_idx % len(COLORS)]
                    
                    for trace in add_traces(df_cat, x_axis, y_primary, False, category=category, color=color):
                        fig.add_trace(trace, secondary_y=False) if use_secondary and y_secondary else fig.add_trace(trace)
                    
                    if use_secondary and y_secondary:
                        for trace in add_traces(df_cat, x_axis, y_secondary, True, category=category, color=color):
                            fig.add_trace(trace, secondary_y=True)
            
            else: # Sin filtro por categoría o con agregación global
                for trace in add_traces(df_plot, x_axis, y_primary, False, category=None, color=None):
                    fig.add_trace(trace, secondary_y=False) if use_secondary and y_secondary else fig.add_trace(trace)
                
                if use_secondary and y_secondary:
                    for trace in add_traces(df_plot, x_axis, y_secondary, True, category=None, color=None):
                        fig.add_trace(trace, secondary_y=True)


            # --- Aplicación de Escala Logarítmica ---
            primary_axis_type = 'log' if y_primary_scale == "Logarítmica" else 'linear'
            secondary_axis_type = 'log' if use_secondary and y_secondary_scale == "Logarítmica" else 'linear'
            xaxis_type = 'log' if x_axis_scale == "Logarítmica" else 'linear'
            
            fig.update_layout(
                title=f"Gráfico de Series - {plot_type}", 
                xaxis_title=x_axis, 
                hovermode='x unified', 
                height=600, 
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05),
                xaxis=dict(type=xaxis_type)
            )

            if use_secondary and y_secondary:
                fig.update_yaxes(title_text=f"Eje Primario ({y_primary_scale})", secondary_y=False, type=primary_axis_type)
                fig.update_yaxes(title_text=f"Eje Secundario ({y_secondary_scale})", secondary_y=True, type=secondary_axis_type)
            else:
                fig.update_yaxes(title_text=f"Valores ({y_primary_scale})", type=primary_axis_type)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Seleccione al menos una variable para el eje Y")

# --- TAB 2: Visualización 3D ---
with tab2:
    st.header("Visualización de Superficie y Puntos (3D)")
    
    col1, col2 = st.columns([1, 3])
    
    if len(numeric_cols) < 3:
        st.warning("Se necesitan al menos 3 columnas numéricas para gráficos 3D.")
    else:
        with col1:
            st.subheader("Configuración")
            plot_type_3d = st.radio("Tipo de Gráfico 3D", ["Puntos", "Línea", "Superficie"])
            
            default_x = numeric_cols[0]
            default_y = numeric_cols[1]
            default_z = numeric_cols[2]
            
            x_axis_3d = st.selectbox("Eje X", numeric_cols, key="x_axis_3d_new", index=numeric_cols.index(default_x) if default_x in numeric_cols else 0)
            y_axis_3d = st.selectbox("Eje Y", numeric_cols, key="y_axis_3d_new", index=numeric_cols.index(default_y) if default_y in numeric_cols else 1)
            z_axis_3d = st.selectbox("Eje Z (Color/Altura)", numeric_cols, key="z_axis_3d_new", index=numeric_cols.index(default_z) if default_z in numeric_cols else 2)
            
            st.markdown("---")
            st.markdown("**Escala de Ejes**")
            col_x_scale, col_y_scale, col_z_scale = st.columns(3)
            x_scale_3d = col_x_scale.radio("Escala X", ["Lineal", "Logarítmica"], key="x_scale_3d_new")
            y_scale_3d = col_y_scale.radio("Escala Y", ["Lineal", "Logarítmica"], key="y_scale_3d_new")
            z_scale_3d = col_z_scale.radio("Escala Z", ["Lineal", "Logarítmica"], key="z_scale_3d_new")
            
            marker_size = 4
            if plot_type_3d == "Puntos":
                marker_size = st.slider("Tamaño de Puntos", 1, 10, 4)
        
        with col2:
            if x_axis_3d and y_axis_3d and z_axis_3d:
                
                x_type_3d = 'log' if x_scale_3d == "Logarítmica" else 'linear'
                y_type_3d = 'log' if y_scale_3d == "Logarítmica" else 'linear'
                z_type_3d = 'log' if z_scale_3d == "Logarítmica" else 'linear'
                
                fig = go.Figure()
                
                if st.session_state.filter_column and st.session_state.selected_categories and not st.session_state.use_binning_global:
                    for cat_idx, category in enumerate(st.session_state.selected_categories):
                        df_cat = df[df[st.session_state.filter_column] == category].dropna(subset=[x_axis_3d, y_axis_3d, z_axis_3d])
                        if df_cat.empty: continue
                        color = COLORS[cat_idx % len(COLORS)]
                        
                        if plot_type_3d == "Puntos":
                            trace = go.Scatter3d(x=df_cat[x_axis_3d], y=df_cat[y_axis_3d], z=df_cat[z_axis_3d], mode='markers', name=category, marker=dict(size=marker_size, color=color))
                        elif plot_type_3d == "Línea":
                            trace = go.Scatter3d(x=df_cat[x_axis_3d], y=df_cat[y_axis_3d], z=df_cat[z_axis_3d], mode='lines', name=category, line=dict(color=color, width=3))
                        elif plot_type_3d == "Superficie" and len(df_cat) >= 4:
                            xi = np.linspace(df_cat[x_axis_3d].min(), df_cat[x_axis_3d].max(), 50)
                            yi = np.linspace(df_cat[y_axis_3d].min(), df_cat[y_axis_3d].max(), 50)
                            XI, YI = np.meshgrid(xi, yi)
                            ZI = griddata((df_cat[x_axis_3d], df_cat[y_axis_3d]), df_cat[z_axis_3d], (XI, YI), method='linear')
                            trace = go.Surface(x=XI, y=YI, z=ZI, name=category, colorscale='Viridis', opacity=0.8, showscale=False)
                        else: continue 
                        
                        fig.add_trace(trace)
                
                else: # Sin filtro por categoría o con agregación global
                    df_plot = df.dropna(subset=[x_axis_3d, y_axis_3d, z_axis_3d])
                    if df_plot.empty: st.warning("Datos insuficientes después de eliminar valores nulos."); st.stop()
                    
                    if plot_type_3d == "Puntos":
                        trace = go.Scatter3d(x=df_plot[x_axis_3d], y=df_plot[y_axis_3d], z=df_plot[z_axis_3d], mode='markers', 
                                             marker=dict(size=marker_size, color=df_plot[z_axis_3d], colorscale='Viridis', showscale=True, colorbar=dict(title=z_axis_3d)))
                    elif plot_type_3d == "Línea":
                        trace = go.Scatter3d(x=df_plot[x_axis_3d], y=df_plot[y_axis_3d], z=df_plot[z_axis_3d], mode='lines', 
                                             line=dict(color=df_plot[z_axis_3d], colorscale='Viridis', width=3))
                    elif plot_type_3d == "Superficie" and len(df_plot) >= 4:
                        xi = np.linspace(df_plot[x_axis_3d].min(), df_plot[x_axis_3d].max(), 50)
                        yi = np.linspace(df_plot[y_axis_3d].min(), df_plot[y_axis_3d].max(), 50)
                        XI, YI = np.meshgrid(xi, yi)
                        ZI = griddata((df_plot[x_axis_3d], df_plot[y_axis_3d]), df_plot[z_axis_3d], (XI, YI), method='linear')
                        trace = go.Surface(x=XI, y=YI, z=ZI, colorscale='Viridis')
                    else: 
                        st.warning("Datos insuficientes para el gráfico de Superficie. Se necesitan al menos 4 puntos únicos de datos X, Y, Z.")
                        st.stop()
                    
                    fig.add_trace(trace)
                
                fig.update_layout(
                    title=f"Gráfico 3D - {plot_type_3d}", 
                    scene=dict(
                        xaxis_title=f"{x_axis_3d} ({x_scale_3d})",
                        yaxis_title=f"{y_axis_3d} ({y_scale_3d})",
                        zaxis_title=f"{z_axis_3d} ({z_scale_3d})",
                        xaxis=dict(type=x_type_3d),
                        yaxis=dict(type=y_type_3d),
                        zaxis=dict(type=z_type_3d)
                    ), 
                    height=700
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Seleccione las tres variables para los ejes X, Y y Z")

# --- TAB 3: Análisis Descriptivo (EDA) ---
with tab3:
    st.header("Análisis Exploratorio de Datos (EDA)")
    
    if numeric_cols:
        
        st.subheader("Configuración de Visualización")
        y_eda_scale = st.radio("Escala Eje Y (Distribución)", ["Lineal", "Logarítmica"], key="y_eda_scale_radio_new")
        y_eda_type = 'log' if y_eda_scale == "Logarítmica" else 'linear'

        selected_vars = st.multiselect("Seleccione variables para analizar", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
        
        if selected_vars:
            # Estadísticas Descriptivas
            st.subheader("Estadísticas Descriptivas")
            stats_df = df[selected_vars].describe().T
            stats_df['media'] = df[selected_vars].mean()
            stats_df['desviacion'] = df[selected_vars].std()
            
            display_df = pd.DataFrame({
                'Variable': stats_df.index,
                'Mínimo': stats_df['min'],
                'Q1 (25%)': stats_df['25%'],
                'Mediana (50%)': stats_df['50%'],
                'Media': stats_df['media'],
                'Q3 (75%)': stats_df['75%'],
                'Máximo': stats_df['max'],
                'Desviación Std': stats_df['desviacion']
            })
            
            st.dataframe(display_df.style.format({col: '{:.4f}' for col in display_df.columns if col != 'Variable'}), use_container_width=True)
            
            # Boxplots 
            st.subheader(f"Distribución (Boxplot) - Escala {y_eda_scale}")
            if st.session_state.filter_column and st.session_state.selected_categories and not st.session_state.use_binning_global:
                fig = go.Figure()
                for cat_idx, category in enumerate(st.session_state.selected_categories):
                    df_cat = df[df[st.session_state.filter_column] == category]
                    color = COLORS[cat_idx % len(COLORS)]
                    for var in selected_vars:
                        fig.add_trace(go.Box(y=df_cat[var], name=f"{var} - {category}", marker_color=color))
            else: # Agregación activa o sin filtro de categoría
                fig = go.Figure()
                for var in selected_vars:
                    fig.add_trace(go.Box(y=df[var], name=var))
            
            fig.update_layout(
                title="Distribución de Variables Seleccionadas", 
                yaxis_title="Valores", 
                height=500, 
                showlegend=True,
                yaxis=dict(type=y_eda_type)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Histogramas
            st.subheader(f"Histogramas - Escala {y_eda_scale}")
            if st.session_state.filter_column and st.session_state.selected_categories and not st.session_state.use_binning_global:
                for var in selected_vars:
                    fig = go.Figure()
                    for cat_idx, category in enumerate(st.session_state.selected_categories):
                        df_cat = df[df[st.session_state.filter_column] == category]
                        color = COLORS[cat_idx % len(COLORS)]
                        fig.add_trace(go.Histogram(x=df_cat[var], name=category, marker_color=color, opacity=0.7))
                    fig.update_layout(
                        title=f"Distribución de {var}", 
                        barmode='overlay', 
                        height=400,
                        yaxis=dict(type=y_eda_type)
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else: # Agregación activa o sin filtro de categoría
                cols_per_row = 3
                num_vars = len(selected_vars)
                num_rows = (num_vars + cols_per_row - 1) // cols_per_row
                
                fig = make_subplots(rows=num_rows, cols=cols_per_row, subplot_titles=selected_vars)
                
                for idx, var in enumerate(selected_vars):
                    row = idx // cols_per_row + 1
                    col = idx % cols_per_row + 1
                    fig.add_trace(go.Histogram(x=df[var], name=var, showlegend=False), row=row, col=col)
                
                fig.update_layout(height=300 * num_rows, title_text="Distribución de Frecuencias")
                
                for i in range(1, num_rows * cols_per_row + 1):
                    fig.update_yaxes(type=y_eda_type, row=i, col='*')
                    
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay columnas numéricas disponibles para análisis descriptivo.")

# --- TAB 4: Correlación de Pearson ---
with tab4:
    st.header("Matriz de Correlación de Pearson")
    
    if len(numeric_cols) >= 2:
        corr_vars = st.multiselect("Seleccione variables para correlación", numeric_cols, default=numeric_cols, key="corr_vars_new")
        
        if len(corr_vars) >= 2:
            if st.session_state.filter_column and st.session_state.selected_categories and not st.session_state.use_binning_global:
                for cat_idx, category in enumerate(st.session_state.selected_categories):
                    st.subheader(f"Categoría: {category}")
                    df_cat = df[df[st.session_state.filter_column] == category][corr_vars]
                    corr_matrix = df_cat.corr(method='pearson')
                    
                    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu', zmid=0, text=corr_matrix.values, texttemplate='%{text:.3f}', textfont={"size": 10}, colorbar=dict(title="Correlación")))
                    fig.update_layout(title=f"Matriz de Correlación - {category}", height=600, xaxis={'side': 'bottom'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander(f"Ver Tabla de Correlación - {category}"):
                        st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1).format("{:.3f}"), use_container_width=True)
            else: # Agregación activa o sin filtro de categoría
                corr_matrix = df[corr_vars].corr(method='pearson')
                
                fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu', zmid=0, text=corr_matrix.values, texttemplate='%{text:.3f}', textfont={"size": 10}, colorbar=dict(title="Correlación")))
                fig.update_layout(title="Matriz de Correlación de Pearson", height=600, xaxis={'side': 'bottom'})
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Valores de Correlación")
                st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1).format("{:.3f}"), use_container_width=True)
            
            corr_pairs = []
            cols = corr_matrix.columns
            for i in range(len(cols)):
                for j in range(i+1, len(cols)):
                    corr_pairs.append({'Variable 1': cols[i], 'Variable 2': cols[j], 'Correlación': corr_matrix.iloc[i, j], 'Correlación Abs': abs(corr_matrix.iloc[i, j])})
            
            corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('Correlación Abs', ascending=False).head(10)
            st.subheader("Correlaciones más Fuertes (Top 10)")
            st.dataframe(corr_pairs_df[['Variable 1', 'Variable 2', 'Correlación']].style.format({'Correlación': "{:.4f}"}), use_container_width=True)
        else:
            st.info("Seleccione al menos 2 variables para calcular correlación.")
    else:
        st.warning("Se necesitan al menos 2 columnas numéricas para calcular correlación.")

# ----------------------------------------
# --- TAB 5: Modelo Predictivo (DNN/BNN) ---
# ----------------------------------------

with tab5:
    st.header("Modelado Predictivo con Redes Neuronales (DNN / BNN)")
    st.markdown("Seleccione el modelo: **DNN** (Determinista) para una predicción única, o **BNN** (Bayesiana, usa MC-Dropout) para predicciones con intervalo de confianza (IC).")
    
    # Recalcular columnas disponibles para ML
    numeric_cols_ml = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols_ml = [col for col in df.columns if df[col].dtype not in unsupported_dtypes]
    
    if len(numeric_cols_ml) < 1 or len(all_cols_ml) < 2:
        st.warning("Se necesitan más variables numéricas y/o no-numéricas para el modelado.")
    else:
        # --- Configuración de Datos ---
        st.subheader("1. Configuración de Datos y Preprocesamiento")
        
        col_y, col_scale = st.columns(2)
        
        # 5.1. Selección de Y
        target_col = col_y.selectbox("Variable Objetivo (Y - Avance Estimado)", numeric_cols_ml, key="target_ml_new")
        
        # 5.2. Selección de X
        exogenous_options = [c for c in all_cols_ml if c != target_col]
        default_exogenous = [c for c in exogenous_options if c in numeric_cols_ml][:min(5, len(exogenous_options))]

        exogenous_cols = st.multiselect(
            "Variables Predictoras (X)", 
            exogenous_options,
            default=default_exogenous
        )

        if not exogenous_cols:
            st.warning("Seleccione al menos una variable predictora (X).")
        else:
            X_df_prep = df[exogenous_cols].copy()
            numeric_features = X_df_prep.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = X_df_prep.select_dtypes(include=['object']).columns.tolist()
            
            # 5.3. Escalado (Para X numéricas)
            scaler_choice = col_scale.selectbox(
                "Escalador Numérico para X", 
                ["StandardScaler", "MinMaxScaler", "RobustScaler", "Ninguno"],
                key="scaler_choice_ml_new"
            )

            # --- Configuración del Modelo (HIPERPARÁMETROS) ---
            st.divider()
            st.subheader("2. Arquitectura y Configuración de Hiperparámetros")
            
            col_model, col_arc = st.columns(2)
            
            # Nuevo: Selección de Modelo
            model_type_choice = col_model.selectbox(
                "Tipo de Red Neuronal", 
                ['DNN (Determinista)', 'BNN (MC-Dropout)'], 
                key="model_type_choice_new",
                help="DNN: Predicción única. BNN: Predicción con intervalo de confianza."
            )
            
            # Arquitectura Dinámica (Nuevo)
            hidden_layers_str = col_arc.text_input(
                "Arquitectura Capas Ocultas (separadas por coma)", 
                value="64, 32", 
                key="dnn_hidden_layers_str",
                help="Defina el número de neuronas en cada capa oculta. Ej: 128, 64, 32"
            )
            try:
                hidden_layers_config = [int(x.strip()) for x in hidden_layers_str.split(',') if x.strip() and x.strip().isdigit()]
                if not hidden_layers_config: hidden_layers_config = [64, 32]
            except:
                 hidden_layers_config = [64, 32]
                 st.error("Error en la conversión de la arquitectura. Usando 64, 32.")

            st.info(f"Arquitectura Seleccionada: {len(hidden_layers_config)} Capas con tamaños: {hidden_layers_config}")
            
            
            # HIPERPARÁMETROS ADICIONALES
            
            # Col 1: Activación y Dropout
            col_act, col_drop = st.columns(2)
            activation_func = col_act.selectbox("Función de Activación", list(ACTIVATIONS.keys()), index=0, key="dnn_act")
            dropout_rate = col_drop.slider("Tasa de Dropout (Activo en BNN)", 0.0, 0.5, 0.1, 0.05, key="dnn_drop")
            
            # Col 2: Optimizador y Weight Decay
            col_opt, col_wd = st.columns(2)
            optimizer_choice = col_opt.selectbox("Optimizador", ['Adam', 'SGD', 'RMSprop'], index=0, key="dnn_opt")
            weight_decay = col_wd.number_input("Weight Decay (L2 Reg.)", min_value=0.0, max_value=0.1, value=0.0, format="%.5f", key="dnn_wd", help="Añade penalización L2 a los pesos para reducir el sobreajuste.")
            
            # Col 3: Momentum e Inicialización
            col_mom, col_init = st.columns(2)
            momentum = col_mom.slider("Momentum (solo para SGD/RMSprop)", 0.0, 0.99, 0.9, 0.05, key="dnn_mom")
            initializer_method = col_init.selectbox("Método de Inicialización", list(INITIALIZERS.keys()), index=0, key="dnn_init")
            
            # Col 4: Scheduler y Función de Pérdida
            col_sch, col_loss = st.columns(2)
            scheduler_choice = col_sch.selectbox("Scheduler de Tasa de Aprendizaje", ['Ninguno', 'StepLR'], index=0, key="dnn_sch", help="Ajusta la tasa de aprendizaje durante el entrenamiento.")
            loss_fn = col_loss.selectbox("Función de Pérdida", ['MSELoss', 'MAELoss'], index=0, key="dnn_loss", help="El Mean Squared Error es el estándar para regresión.")
            
            # Parámetros Específicos de BNN
            if model_type_choice == 'BNN (MC-Dropout)':
                st.markdown("---")
                st.subheader("Configuración Específica BNN (Incertidumbre)")
                col_mc, col_ic = st.columns(2)
                num_monte_carlo_samples = col_mc.number_input(
                    "Muestras Monte Carlo (N) para IC", 
                    min_value=10, value=100, step=10, 
                    key="bnn_mc_samples",
                    help="Número de inferencias con dropout activo para estimar la distribución de predicción y el IC."
                )
                confidence_level = col_ic.slider("Nivel de Confianza (%)", 50, 99, 95) / 100
            else:
                num_monte_carlo_samples = 1 # Irrelevante para DNN
                confidence_level = 0.95


            # Parámetros de Entrenamiento
            st.markdown("---")
            st.subheader("3. Parámetros de Entrenamiento (PyTorch)")
            
            col_ep, col_bs, col_lr = st.columns(3)
            epochs = col_ep.number_input("Épocas de Entrenamiento", min_value=10, value=100, step=10, key="dnn_epochs_new")
            batch_size = col_bs.slider("Tamaño del Lote (Batch Size)", 1, 128, 32, key="dnn_bs_new")
            learning_rate = col_lr.number_input("Tasa de Aprendizaje (Learning Rate)", min_value=1e-5, max_value=0.1, value=0.001, format="%.5f", key="dnn_lr_new")
            
            st.markdown(f"**Aceleración de Hardware:** {DEVICE.upper()}")
            
            # Parámetros de Optimización y División
            st.markdown("---")
            st.subheader("4. División de Datos")
            col_split, col_holdout = st.columns(2)
            test_size = col_split.slider("Tamaño del Conjunto de Prueba (%)", 5, 50, 20) / 100
            holdout_pct = col_holdout.slider("Porcentaje de Datos de Validación Final ('Hold-out')", 0, 50, 10) / 100
            
            # Configuración de Visualización
            st.markdown("---")
            st.subheader("5. Configuración de Visualización")
            ml_plot_y_scale = st.radio("Escala Eje Y (Predicción)", ["Lineal", "Logarítmica"], key="ml_plot_y_scale_radio_new")
            ml_y_type = 'log' if ml_plot_y_scale == "Logarítmica" else 'linear'

            # --- Botón de Ejecución ---
            st.divider()
            if st.button(f"Entrenar Modelo ({model_type_choice})", use_container_width=True):
                
                # 1. Preparar Y
                Y_data = df[target_col].copy()
                target_name_internal = target_col
                
                data_ml = X_df_prep.copy()
                data_ml[target_name_internal] = Y_data.values 
                data_ml_clean = data_ml.dropna(subset=exogenous_cols + [target_name_internal])
                
                if data_ml_clean.empty:
                    st.error("No quedan datos válidos para el entrenamiento después de eliminar valores faltantes.")
                    st.stop()
                    
                # 2. Separar datos de Hold-out (X_model, Y_model)
                if holdout_pct > 0 and len(data_ml_clean) * (1-holdout_pct) >= 2:
                    df_model, df_holdout = train_test_split(data_ml_clean, test_size=holdout_pct, random_state=42)
                else:
                    df_model = data_ml_clean
                    df_holdout = None

                X = df_model[exogenous_cols]
                Y = df_model[target_name_internal]
                
                # 3. División de datos (Train para fit, Test para evaluación final)
                if len(df_model) * (1-test_size) >= 2 and len(df_model) * test_size >= 1:
                    X_train, X_test, Y_train, Y_test = train_test_split(
                        X, Y, test_size=test_size, random_state=42
                    )
                else:
                    st.error("Insuficientes datos para la división Train/Test seleccionada.")
                    st.stop()
                
                st.info(f"Datos de Entrenamiento: {len(X_train)} | Datos de Prueba (Test): {len(X_test)} | Datos de Validación Final (Hold-out): {len(df_holdout) if df_holdout is not None else 0}")
                
                # 4. Preprocesamiento (Pipeline)
                if scaler_choice == "StandardScaler":
                    numeric_transformer = StandardScaler()
                elif scaler_choice == "MinMaxScaler":
                    numeric_transformer = MinMaxScaler()
                elif scaler_choice == "RobustScaler":
                    numeric_transformer = RobustScaler()
                else:
                    numeric_transformer = 'passthrough'

                categorical_transformer = OneHotEncoder(handle_unknown='ignore')
                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                        ('cat', categorical_transformer, categorical_features)
                    ],
                    remainder='passthrough'
                )
                
                # 5. Definición del Modelo PyTorch - Actualizado
                pytorch_regressor = PyTorchRegressor(
                    hidden_layers=hidden_layers_config, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    learning_rate=learning_rate, 
                    device=DEVICE,
                    dropout_rate=dropout_rate,
                    activation_func=activation_func,
                    optimizer_choice=optimizer_choice,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    initializer_method=initializer_method,
                    loss_fn=loss_fn,
                    scheduler_choice=scheduler_choice,
                    model_type=model_type_choice, # Nuevo
                    num_monte_carlo_samples=num_monte_carlo_samples # Nuevo
                )
                
                start_time = time.time()
                with st.spinner(f"Entrenando la Red Neuronal ({model_type_choice}) en {DEVICE.upper()}..."):
                    
                    X_train_processed = preprocessor.fit_transform(X_train)
                    X_test_processed = preprocessor.transform(X_test)
                    
                    pytorch_regressor.fit(X_train_processed, Y_train) 
                    
                end_time = time.time()
                
                st.success(f"Entrenamiento de **{model_type_choice}** completado en {end_time - start_time:.2f} segundos en {DEVICE.upper()}.")
                
                # Guardar el modelo entrenado y los parámetros para la simulación
                st.session_state.trained_regressor = pytorch_regressor
                st.session_state.trained_preprocessor = preprocessor
                st.session_state.target_name = target_name_internal
                st.session_state.exogenous_cols = exogenous_cols
                st.session_state.numeric_features_ml = numeric_features
                st.session_state.categorical_features_ml = categorical_features
                st.session_state.trained_model_type = model_type_choice # Guardar tipo de modelo
                st.session_state.trained_confidence_level = confidence_level # Guardar nivel de confianza
                
                # --- Cálculo y Almacenamiento de Importancia de Variables (Mismo código) ---
                
                # 1. Obtener nombres de features post-preprocesamiento (incluyendo OHE)
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except:
                    num_features_out = X_train_processed.shape[1] if hasattr(X_train_processed, 'shape') else 0
                    feature_names = [f'feature_{i}' for i in range(num_features_out)]

                # 2. Extraer pesos de la primera capa (Pesos w de las conexiones Input -> Hidden1)
                first_layer_weights = pytorch_regressor.model.model[0].weight.data.cpu().numpy()
                
                # 3. Calcular la importancia como la norma L2 (magnitud) de los pesos por cada entrada
                importance = np.linalg.norm(first_layer_weights, axis=0)
                
                # 4. Crear DataFrame de Importancia
                importance_df = pd.DataFrame({
                    'Variable (Preprocesada)': feature_names,
                    'Importancia (Norma L2 de Pesos)': importance
                }).sort_values(by='Importancia (Norma L2 de Pesos)', ascending=False).reset_index(drop=True)

                st.session_state.feature_importance_df = importance_df
                
                # 6. Evaluación y Métricas (Adaptadas para BNN)
                
                def evaluate_model(model, X_data_processed, Y_true, name, model_type, confidence_level):
                    
                    if model_type == 'BNN (MC-Dropout)':
                        # Predicción para BNN (Devuelve DF de resultados)
                        Y_pred_results = model.predict(X_data_processed, return_uncertainty=True, confidence_level=confidence_level)
                        Y_pred = Y_pred_results['Y_Predicho_Media'].values # Usamos la media para las métricas
                        Y_pred_df = pd.DataFrame({
                            'Y_Media': Y_pred,
                            'Y_Inferior': Y_pred_results['Y_Predicho_Inferior'].values,
                            'Y_Superior': Y_pred_results['Y_Predicho_Superior'].values,
                        }, index=Y_true.index)
                        
                    else:
                        # Predicción para DNN (Devuelve array)
                        Y_pred = model.predict(X_data_processed, return_uncertainty=False)
                        Y_pred_df = pd.DataFrame({
                            'Y_Media': Y_pred,
                        }, index=Y_true.index)


                    mse = mean_squared_error(Y_true, Y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(Y_true, Y_pred)
                    r2 = r2_score(Y_true, Y_pred)
                    
                    st.markdown(f"**Métricas del Conjunto de {name}:**")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Coeficiente de Determinación (R²)", f"{r2:.4f}")
                    col2.metric("RMSE", f"{rmse:.4f}")
                    col3.metric("MAE", f"{mae:.4f}")
                    col4.metric("MSE", f"{mse:.4f}")
                    
                    return Y_pred_df

                st.subheader(f"Métricas del Modelo: {model_type_choice}")
                
                Y_pred_train_df = evaluate_model(pytorch_regressor, X_train_processed, Y_train, "Entrenamiento", model_type_choice, confidence_level)
                st.divider()

                Y_pred_test_df = evaluate_model(pytorch_regressor, X_test_processed, Y_test, "Prueba (Generalización)", model_type_choice, confidence_level)
                st.divider()
                
                # Muestra la importancia de variables justo después de las métricas
                st.subheader("Peso de Variables (Feature Importance)")
                st.markdown("El peso se calcula como la magnitud de los pesos de la primera capa de la red neuronal. Indica la influencia inicial de la variable en la predicción.")
                st.dataframe(st.session_state.feature_importance_df, use_container_width=True)

                # 7. Visualización de Predicciones (Test) - Actualizado para BNN
                st.subheader(f"Visualización de Predicciones vs Reales (Prueba) - Escala Y: {ml_plot_y_scale}")

                results_df = pd.DataFrame({
                    target_name_internal + " - Real": Y_test,
                    target_name_internal + " - Predicho (Media)": Y_pred_test_df['Y_Media'],
                }).sort_index().reset_index()

                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    x=results_df.index, y=results_df[target_name_internal + " - Real"], 
                    mode='lines+markers', name="Valor Real", line=dict(color='blue')
                ))
                
                if model_type_choice == 'BNN (MC-Dropout)':
                    # Agregar IC para BNN
                    results_df['Límite Inferior'] = Y_pred_test_df['Y_Inferior']
                    results_df['Límite Superior'] = Y_pred_test_df['Y_Superior']
                    
                    fig_comp.add_trace(go.Scatter(
                        x=results_df.index.tolist() + results_df.index.tolist()[::-1],
                        y=results_df['Límite Superior'].tolist() + results_df['Límite Inferior'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'IC {confidence_level*100:.0f}%'
                    ))
                    
                    fig_comp.add_trace(go.Scatter(
                        x=results_df.index, y=results_df[target_name_internal + " - Predicho (Media)"], 
                        mode='lines', name="Valor Predicho (Media)", line=dict(color='red', dash='dash', width=2)
                    ))
                else:
                    fig_comp.add_trace(go.Scatter(
                        x=results_df.index, y=results_df[target_name_internal + " - Predicho (Media)"], 
                        mode='lines', name="Valor Predicho", line=dict(color='red', dash='dash', width=2)
                    ))
                
                fig_comp.update_layout(
                    title=f"Predicción vs Valor Real en Prueba ({target_name_internal})",
                    xaxis_title="Índice de la Muestra",
                    yaxis_title=target_name_internal,
                    height=500,
                    yaxis=dict(type=ml_y_type)
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # 8. Predicciones Hold-out (Tabla y Gráfico) - Actualizado para BNN
                if df_holdout is not None and not df_holdout.empty:
                    st.subheader(f"Predicciones en Datos de Validación Final ({len(df_holdout)} filas)")
                    X_holdout = df_holdout[exogenous_cols]
                    Y_real_holdout = df_holdout[target_name_internal]
                    
                    X_holdout_processed = preprocessor.transform(X_holdout)
                    
                    # Predicción con incertidumbre o no
                    Y_pred_holdout_df = evaluate_model(pytorch_regressor, X_holdout_processed, Y_real_holdout, "Validación Final", model_type_choice, confidence_level)
                    
                    holdout_results = df_holdout.copy()
                    holdout_results['Y_Predicho_Media'] = Y_pred_holdout_df['Y_Media'].values
                    
                    # Columnas para mostrar en tabla
                    cols_to_show = [target_name_internal, 'Y_Predicho_Media']
                    if model_type_choice == 'BNN (MC-Dropout)':
                        holdout_results['Y_Predicho_Inferior'] = Y_pred_holdout_df['Y_Inferior'].values
                        holdout_results['Y_Predicho_Superior'] = Y_pred_holdout_df['Y_Superior'].values
                        cols_to_show.extend(['Y_Predicho_Inferior', 'Y_Predicho_Superior'])
                        
                    cols_to_show.extend(exogenous_cols)
                    
                    format_dict = {col: "{:.4f}" for col in holdout_results.columns if holdout_results[col].dtype in ['float64', 'float32']}

                    st.dataframe(
                        holdout_results[cols_to_show].head(10)
                        .style.format(format_dict), 
                        use_container_width=True
                    )
                    
                    # GRÁFICO DEL HOLD-OUT - Actualizado para BNN
                    st.markdown(f"##### Comparativa Gráfica Validación Final - Escala Y: {ml_plot_y_scale}")
                    holdout_results_plot = holdout_results[[target_name_internal, 'Y_Predicho_Media']].reset_index()
                    
                    fig_holdout = go.Figure()
                    fig_holdout.add_trace(go.Scatter(
                        x=holdout_results_plot.index, y=holdout_results_plot[target_name_internal], 
                        mode='lines+markers', name="Real (Validación)", line=dict(color='green')
                    ))
                    
                    if model_type_choice == 'BNN (MC-Dropout)':
                        # Agregar IC para BNN
                        fig_holdout.add_trace(go.Scatter(
                            x=holdout_results_plot.index.tolist() + holdout_results_plot.index.tolist()[::-1],
                            y=Y_pred_holdout_df['Y_Superior'].tolist() + Y_pred_holdout_df['Y_Inferior'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(255, 165, 0, 0.1)', # Naranja claro
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f'IC {confidence_level*100:.0f}%'
                        ))
                        fig_holdout.add_trace(go.Scatter(
                            x=holdout_results_plot.index, y=holdout_results_plot['Y_Predicho_Media'], 
                            mode='lines', name="Predicho (Media)", line=dict(color='orange', dash='dash', width=2)
                        ))
                    else:
                        fig_holdout.add_trace(go.Scatter(
                            x=holdout_results_plot.index, y=holdout_results_plot['Y_Predicho_Media'], 
                            mode='lines', name="Predicho (Validación)", line=dict(color='orange', dash='dash', width=2)
                        ))
                    
                    fig_holdout.update_layout(
                        title=f"Predicción vs Valor Real en Validación Final ({target_name_internal})",
                        xaxis_title="Índice de la Muestra",
                        yaxis_title=target_name_internal,
                        height=500,
                        yaxis=dict(type=ml_y_type)
                    )
                    st.plotly_chart(fig_holdout, use_container_width=True)


                    # Descarga: Incluir columnas IC si es BNN
                    download_df = holdout_results[[target_name_internal] + cols_to_show[1:]]
                    csv = download_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar Predicciones Validación Final",
                        data=csv,
                        file_name=f'predicciones_{model_type_choice}_avance.csv',
                        mime='text/csv',
                    )
                else:
                    st.info("No se generaron datos de Validación Final ('Hold-out') para esta corrida.")
            
            # 9. Sección de Simulación (SOLO se muestra si el modelo fue entrenado)
            st.markdown("---")
            st.subheader("6. Simulación y Predicción de Nuevos Escenarios")
            
            if st.session_state.trained_regressor is not None:
                st.success(f"Modelo **{st.session_state.trained_model_type}** listo para predicción de nuevos escenarios.")
                
                sim_mode = st.radio("Método de Ingreso de Datos de Simulación", ["Ingreso Manual de Escenarios", "Carga de Archivo Excel/CSV"], key="sim_mode")
                
                
                if st.session_state.trained_model_type == 'BNN (MC-Dropout)':
                    st.info(f"El modelo BNN entregará predicciones con Intervalo de Confianza del {st.session_state.trained_confidence_level*100:.0f}%.")
                    
                
                if sim_mode == "Ingreso Manual de Escenarios":
                    num_scenarios = st.number_input("Número de Escenarios a Simular (N)", 1, 10, 3, key="num_scenarios")
                    
                    st.markdown(f"**Ingrese los valores para las {len(st.session_state.exogenous_cols)} variables predictoras:**")
                    
                    sim_data = {}
                    
                    # Generar columnas de Streamlit dinámicamente
                    cols_per_row = 3
                    
                    for i in range(num_scenarios):
                        st.markdown(f"##### Escenario {i+1}")
                        scenario_data = {}
                        
                        # Usar columns() para organizar los inputs
                        cols_input = st.columns(cols_per_row)
                        
                        for idx, col in enumerate(st.session_state.exogenous_cols):
                            col_index = idx % cols_per_row
                            
                            # Obtener tipo de variable para el input correcto
                            is_numeric = col in st.session_state.numeric_features_ml
                            
                            with cols_input[col_index]:
                                if is_numeric:
                                    # Para variables numéricas, usar number_input
                                    val = st.number_input(
                                        f"{col}", 
                                        key=f"sim_input_{i}_{col}",
                                        value=df[col].mean() if col in df.columns else 0.0
                                    )
                                    scenario_data[col] = val
                                else:
                                    # Para categóricas, usar selectbox con valores únicos del training
                                    if col in st.session_state.data.columns:
                                        # Usamos el DataFrame original para obtener categorías, ya que el DF binned no las tiene
                                        unique_cats = st.session_state.data[col].dropna().unique().tolist()
                                        if not unique_cats:
                                            st.warning(f"No hay valores únicos para {col} en los datos. Usando 'N/A'.")
                                            unique_cats = ['N/A']
                                        val = st.selectbox(
                                            f"{col}", 
                                            unique_cats,
                                            key=f"sim_input_{i}_{col}"
                                        )
                                        scenario_data[col] = val
                                    else:
                                        # Caso donde la columna categórica no está en el DF de trabajo (ej: fue agregada)
                                        st.text_input(f"{col} (Categórica)", value="Valor Desconocido", key=f"sim_input_{i}_{col}")
                                        scenario_data[col] = "Valor Desconocido" 
                                        
                        sim_data[f"Escenario {i+1}"] = scenario_data
                    
                    # Botón para ejecutar la predicción manual
                    if st.button(f"Predecir {num_scenarios} Escenarios", key="predict_manual_btn"):
                        new_X_df = pd.DataFrame.from_dict(sim_data, orient='index')
                        
                        # Asegurarse de que el orden de las columnas sea el correcto
                        new_X_df = new_X_df[st.session_state.exogenous_cols]
                        
                        # Ejecutar la predicción
                        new_Y_pred, prediction_df = make_prediction(
                            new_X_df, 
                            st.session_state.target_name, 
                            st.session_state.trained_model_type,
                            st.session_state.trained_confidence_level
                        )
                        
                        if prediction_df is not None:
                            st.markdown("#### Resultados de la Simulación Manual")
                            
                            # Formato de la tabla para predicción y IC (si aplica)
                            format_dict_sim = {col: "{:.4f}" for col in prediction_df.columns if col.startswith("Predicción") or col.startswith("Límite")}
                            st.dataframe(prediction_df.style.format(format_dict_sim), use_container_width=True)
                            
                            # Gráfico de predicción
                            plot_prediction_graph(
                                prediction_df, 
                                st.session_state.target_name, 
                                ml_y_type, 
                                st.session_state.trained_model_type,
                                st.session_state.trained_confidence_level
                            )
                            
                            st.session_state['last_sim_df'] = prediction_df # Guardar para posible descarga

                
                elif sim_mode == "Carga de Archivo Excel/CSV":
                    uploaded_sim_file = st.file_uploader("Subir archivo Excel/CSV con variables (X)", type=['xlsx', 'xls', 'csv'], key="sim_file_uploader")

                    st.info(f"El archivo debe contener exactamente las siguientes columnas (variables predictoras): **{st.session_state.exogenous_cols}**")

                    if uploaded_sim_file:
                        try:
                            if uploaded_sim_file.name.endswith('csv'):
                                new_X_df = pd.read_csv(uploaded_sim_file)
                            else:
                                new_X_df = pd.read_excel(uploaded_sim_file)
                            
                            # Validar columnas
                            missing_cols = [col for col in st.session_state.exogenous_cols if col not in new_X_df.columns]
                            
                            if missing_cols:
                                st.error(f"El archivo subido está incompleto. Faltan las siguientes columnas: {missing_cols}")
                            else:
                                # Asegurar el orden correcto de las columnas y solo las necesarias
                                new_X_df = new_X_df[st.session_state.exogenous_cols]
                                
                                # Ejecutar la predicción
                                new_Y_pred, prediction_df = make_prediction(
                                    new_X_df, 
                                    st.session_state.target_name, 
                                    st.session_state.trained_model_type,
                                    st.session_state.trained_confidence_level
                                )
                                
                                if prediction_df is not None:
                                    st.markdown("#### Resultados de la Predicción por Archivo")
                                    
                                    # Formato de la tabla para predicción y IC (si aplica)
                                    format_dict_sim = {col: "{:.4f}" for col in prediction_df.columns if col.startswith("Predicción") or col.startswith("Límite")}
                                    st.dataframe(prediction_df.style.format(format_dict_sim), use_container_width=True)
                                    
                                    # Gráfico de predicción
                                    plot_prediction_graph(
                                        prediction_df, 
                                        st.session_state.target_name, 
                                        ml_y_type, 
                                        st.session_state.trained_model_type,
                                        st.session_state.trained_confidence_level
                                    )

                                    st.session_state['last_sim_df'] = prediction_df
                                    
                                    # Botón de descarga para las predicciones
                                    csv = prediction_df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="Descargar Predicciones (Archivo)",
                                        data=csv,
                                        file_name=f'predicciones_nuevos_escenarios_{st.session_state.trained_model_type}.csv',
                                        mime='text/csv',
                                    )
                                
                        except Exception as e:
                            st.error(f"Error al procesar el archivo: {e}")
            
            else:
                st.warning("Debe entrenar el modelo (Secciones 1 a 5) antes de poder simular nuevos escenarios.")


# ----------------------------------------
# --- Pie de página ---
# ----------------------------------------
st.divider()
if st.session_state.data is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Filas Originales", len(st.session_state.data))
    with col2:
        status_text = "Filtradas" if not st.session_state.use_binning_global else f"Agrupadas por Intervalo ({st.session_state.agg_method_global})"
        st.metric(f"Filas de Datos de Trabajo ({status_text})", len(df))
    with col3:
        st.metric("Columnas Numéricas de Trabajo", len(numeric_cols))