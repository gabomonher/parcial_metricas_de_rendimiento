# Dashboard de Predicción de Niveles de Obesidad

Este es un dashboard interactivo desarrollado con Dash para predecir niveles de obesidad basado en diferentes características del usuario.

## Características

- Interfaz de usuario intuitiva y responsiva
- Predicción en tiempo real
- Validación de datos de entrada
- Manejo de errores robusto

## Requisitos

- Python 3.10 o superior
- Dependencias listadas en `requirements.txt`

## Instalación

1. Clonar el repositorio:
```bash
git clone [URL_DEL_REPOSITORIO]
cd [NOMBRE_DEL_DIRECTORIO]
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

3. Colocar los archivos del modelo en la raíz del proyecto:
- `modelo_obesidad.pkl`
- `target_encoder.pkl`
- `encoders_dict.pkl`

## Uso

1. Ejecutar la aplicación:
```bash
python app.py
```

2. Abrir el navegador en `http://localhost:8050`

## Estructura del Proyecto

```
mi_dashboard_obesidad/
├── app.py              # Aplicación principal
├── requirements.txt    # Dependencias
├── Procfile           # Configuración para despliegue
├── modelo_obesidad.pkl
├── target_encoder.pkl
└── encoders_dict.pkl
```

## Variables de Entrada

- Edad
- Género
- Altura
- Peso
- Frecuencia de consumo de vegetales (FCVC)
- Número de comidas principales (NCP)
- Consumo de agua diario (CH2O)
- Frecuencia de actividad física (FAF)
- Tiempo usando dispositivos tecnológicos (TUE)
- Consumo de alcohol (CALC)
- Consumo de alimentos altos en calorías (FAVC)
- Monitoreo de calorías (SCC)
- Fumar (SMOKE)
- Historial familiar de sobrepeso
- Comer entre comidas (CAEC)
- Transporte principal (MTRANS)

## Licencia

Este proyecto está bajo la Licencia MIT. 