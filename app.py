import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import joblib
import os # Para construir rutas de archivos de forma robusta

# --- 0. Configuración de rutas a los archivos del modelo ---
# Obtener la ruta del directorio actual donde se ejecuta app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_obesidad.pkl')
TARGET_ENCODER_PATH = os.path.join(BASE_DIR, 'target_encoder.pkl')
FEATURE_ENCODERS_PATH = os.path.join(BASE_DIR, 'encoders_dict.pkl')

# --- 1. Cargar Modelos y Encoders ---
try:
    model = joblib.load(MODEL_PATH)
    target_encoder = joblib.load(TARGET_ENCODER_PATH)
    feature_encoders = joblib.load(FEATURE_ENCODERS_PATH)
    print("Modelos y encoders cargados correctamente.")
except Exception as e:
    print(f"Error crítico cargando archivos del modelo: {e}")
    print(f"Verifica que los archivos .pkl estén en la ruta correcta: {BASE_DIR}")
    # En un entorno de producción, podrías querer que la app falle o muestre un error persistente.
    # Por ahora, definimos como None para que la app pueda arrancar y mostrar el error en la UI.
    model, target_encoder, feature_encoders = None, None, None

# --- 2. Definir las opciones para los Dropdowns (basado en el dataset original) ---
# Estas son las categorías originales ANTES de la codificación.
# El 'value' debe ser el string que luego será transformado por el LabelEncoder.
gender_options = [{'label': 'Masculino', 'value': 'Male'}, {'label': 'Femenino', 'value': 'Female'}]
yes_no_options = [{'label': 'Sí', 'value': 'yes'}, {'label': 'No', 'value': 'no'}]
calc_options = [
    {'label': 'No consume alcohol', 'value': 'no'},
    {'label': 'Consume alcohol a veces', 'value': 'Sometimes'},
    {'label': 'Consume alcohol frecuentemente', 'value': 'Frequently'},
    {'label': 'Consume alcohol siempre', 'value': 'Always'}
]
caec_options = [
    {'label': 'No come entre comidas', 'value': 'no'},
    {'label': 'Come entre comidas a veces', 'value': 'Sometimes'},
    {'label': 'Come entre comidas frecuentemente', 'value': 'Frequently'},
    {'label': 'Come entre comidas siempre', 'value': 'Always'}
]
mtrans_options = [
    {'label': 'Automóvil', 'value': 'Automobile'},
    {'label': 'Motocicleta', 'value': 'Motorbike'},
    {'label': 'Bicicleta', 'value': 'Bike'},
    {'label': 'Transporte Público', 'value': 'Public_Transportation'},
    {'label': 'Caminando', 'value': 'Walking'}
]

# Lista de columnas en el orden que el modelo espera (basado en el notebook original)
# Esta lista DEBE coincidir exactamente con las características usadas durante el entrenamiento.
expected_cols = ['Age', 'Gender', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC', 'NCP', 'SCC', 'SMOKE',
                 'CH2O', 'family_history_with_overweight', 'FAF', 'TUE', 'CAEC', 'MTRANS']

# --- 3. Inicializar la App de Dash ---
app = dash.Dash(__name__)
server = app.server # Exponer el servidor Flask para Gunicorn

# --- 4. Definir el Layout del Dashboard ---
app.layout = html.Div([
    html.H1("Predicción de Niveles de Obesidad (POC)", style={'textAlign': 'center', 'marginBottom': '30px'}),

    html.Div([
        # Columna Izquierda de Inputs
        html.Div([
            html.Label("Edad:", style={'fontWeight': 'bold'}),
            dcc.Input(id='age', type='number', value=25, className="dash-input", style={'marginBottom': '10px', 'width': '100%'}),

            html.Label("Altura (metros, ej: 1.75):", style={'fontWeight': 'bold'}),
            dcc.Input(id='height', type='number', value=1.70, step=0.01, className="dash-input", style={'marginBottom': '10px', 'width': '100%'}),

            html.Label("Peso (kg, ej: 70.5):", style={'fontWeight': 'bold'}),
            dcc.Input(id='weight', type='number', value=70, step=0.1, className="dash-input", style={'marginBottom': '10px', 'width': '100%'}),

            html.Label("FCVC (Frecuencia de consumo de vegetales, 1-nunca, 2-a veces, 3-siempre):", style={'fontWeight': 'bold'}),
            dcc.Slider(id='fcvc', min=1, max=3, step=1, value=2, marks={i: str(i) for i in range(1, 4)}, className="dash-slider", included=False),
            html.Br(),

            html.Label("NCP (Número de comidas principales, 1-4):", style={'fontWeight': 'bold'}),
            dcc.Slider(id='ncp', min=1, max=4, step=1, value=3, marks={i: str(i) for i in range(1, 5)}, className="dash-slider", included=False),
            html.Br(),

            html.Label("CH2O (Consumo de agua diario - Litros, 1-<1L, 2=1-2L, 3=>2L):", style={'fontWeight': 'bold'}),
            dcc.Slider(id='ch2o', min=1, max=3, step=1, value=2, marks={i: str(i) for i in range(1, 4)}, className="dash-slider", included=False),
            html.Br(),

            html.Label("FAF (Frecuencia de actividad física, 0-ninguna, 1=1-2 días, 2=2-4 días, 3=4-5 días):", style={'fontWeight': 'bold'}),
            dcc.Slider(id='faf', min=0, max=3, step=1, value=1, marks={i: str(i) for i in range(0, 4)}, className="dash-slider", included=False),
            html.Br(),

            html.Label("TUE (Tiempo usando dispositivos tecnológicos - Horas, 0=0-2h, 1=3-5h, 2=>5h):", style={'fontWeight': 'bold'}),
            dcc.Slider(id='tue', min=0, max=2, step=1, value=1, marks={i: str(i) for i in range(0, 3)}, className="dash-slider", included=False),
            html.Br(),

        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '2%'}),

        # Columna Derecha de Inputs
        html.Div([
            html.Label("Género:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='gender', options=gender_options, value='Male', className="dash-dropdown", style={'marginBottom': '10px'}),

            html.Label("¿Consume alcohol? (CALC):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='calc', options=calc_options, value='Sometimes', className="dash-dropdown", style={'marginBottom': '10px'}),

            html.Label("¿Consume alimentos altos en calorías frecuentemente? (FAVC):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='favc', options=yes_no_options, value='yes', className="dash-dropdown", style={'marginBottom': '10px'}),

            html.Label("¿Monitorea el consumo de calorías? (SCC):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='scc', options=yes_no_options, value='no', className="dash-dropdown", style={'marginBottom': '10px'}),

            html.Label("¿Fuma? (SMOKE):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='smoke', options=yes_no_options, value='no', className="dash-dropdown", style={'marginBottom': '10px'}),

            html.Label("¿Historial familiar de sobrepeso? (family_history_with_overweight):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='family_history', options=yes_no_options, value='yes', className="dash-dropdown", style={'marginBottom': '10px'}),

            html.Label("¿Come algo entre comidas? (CAEC):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='caec', options=caec_options, value='Sometimes', className="dash-dropdown", style={'marginBottom': '10px'}),

            html.Label("Transporte principal (MTRANS):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='mtrans', options=mtrans_options, value='Public_Transportation', className="dash-dropdown", style={'marginBottom': '20px'}),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '30px'}),

    html.Div([
        html.Button('Predecir Nivel de Obesidad', id='predict-button', n_clicks=0, style={'backgroundColor': '#007bff', 'color': 'white', 'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
    ], style={'textAlign': 'center'}),

    html.Div(id='prediction-output', style={'fontSize': 22, 'marginTop': '30px', 'textAlign': 'center', 'fontWeight': 'bold', 'padding': '15px', 'border': '1px dashed #ccc', 'borderRadius': '5px'})
], style={'maxWidth': '800px', 'margin': 'auto', 'fontFamily': 'Arial, sans-serif'})

# --- 5. Definir Callbacks para la interactividad ---
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('age', 'value'), State('gender', 'value'), State('height', 'value'), State('weight', 'value'),
     State('calc', 'value'), State('favc', 'value'), State('fcvc', 'value'), State('ncp', 'value'),
     State('scc', 'value'), State('smoke', 'value'), State('ch2o', 'value'), State('family_history', 'value'),
     State('faf', 'value'), State('tue', 'value'), State('caec', 'value'), State('mtrans', 'value')]
)
def update_prediction(n_clicks, age, gender, height, weight, calc, favc, fcvc, ncp, scc,
                      smoke, ch2o, family_history, faf, tue, caec, mtrans):
    if n_clicks == 0:
        return "Por favor, ingresa los datos y presiona predecir."

    if not all([model, target_encoder, feature_encoders]):
        return "Error: No se pudieron cargar los archivos del modelo. Revisa los logs del servidor."

    # Validar que todos los inputs tengan valor (básico)
    inputs_dict_state = {
        'Age': age, 'Gender': gender, 'Height': height, 'Weight': weight, 'CALC': calc, 'FAVC': favc,
        'FCVC': fcvc, 'NCP': ncp, 'SCC': scc, 'SMOKE': smoke, 'CH2O': ch2o,
        'family_history_with_overweight': family_history, 'FAF': faf, 'TUE': tue, 'CAEC': caec, 'MTRANS': mtrans
    }
    if any(value is None for value in inputs_dict_state.values()):
        return "Error: Todos los campos son requeridos. Por favor, completa toda la información."

    # Crear un DataFrame con los inputs del usuario en el orden correcto
    # y con los tipos de datos correctos.
    try:
        input_data_raw = {
            'Age': [float(age)],
            'Gender': [gender],
            'Height': [float(height)],
            'Weight': [float(weight)],
            'CALC': [calc],
            'FAVC': [favc],
            'FCVC': [float(fcvc)],
            'NCP': [float(ncp)],
            'SCC': [scc],
            'SMOKE': [smoke],
            'CH2O': [float(ch2o)],
            'family_history_with_overweight': [family_history],
            'FAF': [float(faf)],
            'TUE': [float(tue)],
            'CAEC': [caec],
            'MTRANS': [mtrans]
        }
        df_predict = pd.DataFrame(input_data_raw)

        # Aplicar los LabelEncoders guardados a las columnas categóricas
        for col_name, encoder in feature_encoders.items():
            if col_name in df_predict.columns:
                # Convertir la columna a string antes de transformar, por si acaso
                # y para manejar el caso de que el input ya sea un número por error.
                # El encoder espera strings si fue entrenado con strings.
                df_predict[col_name] = df_predict[col_name].astype(str)
                df_predict[col_name] = encoder.transform(df_predict[col_name])
            else:
                print(f"Advertencia: La columna '{col_name}' del encoder no se encontró en los datos de entrada.")

        # Asegurar el orden de las columnas según `expected_cols`
        # Esto es CRUCIAL si el modelo es sensible al orden de las características.
        df_predict = df_predict[expected_cols]

    except Exception as e:
        return f"Error al preprocesar los datos de entrada: {e}. Verifica los valores."

    # Realizar la predicción
    try:
        prediction_encoded = model.predict(df_predict)
        # Decodificar la predicción a la etiqueta original
        prediction_label = target_encoder.inverse_transform(prediction_encoded)
        return f'Predicción: {prediction_label[0]}'
    except Exception as e:
        return f"Error al realizar la predicción: {e}"

# --- 6. Correr la App ---
if __name__ == '__main__':
    # Cuando se despliega, Gunicorn usa 'server'. Para desarrollo local, esto está bien.
    app.run(debug=True) 