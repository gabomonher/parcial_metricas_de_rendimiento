# Dashboard de Predicción de Niveles de Obesidad

Este es un dashboard interactivo desarrollado con **Dash** para predecir niveles de obesidad basado en diferentes características del usuario.

## Características

- Interfaz de usuario intuitiva y responsiva
- Predicción en tiempo real
- Validación de datos de entrada
- Manejo de errores robusto

## Requisitos

- Python 3.10 o superior
- Dependencias listadas en `requirements.txt`

## Instalación Local

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/gabomonher/parcial_metricas_de_rendimiento.git
   cd parcial_metricas_de_rendimiento
   ```

2. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Coloca los archivos del modelo en la raíz del proyecto:**
   - `modelo_obesidad.pkl`
   - `target_encoder.pkl`
   - `encoders_dict.pkl`

4. **Ejecuta la aplicación:**
   ```bash
   python app.py
   ```

5. **Abre tu navegador en:**  
   [http://localhost:8050](http://localhost:8050)

## Despliegue en Render

1. **Sube tu código a un repositorio de GitHub.**
2. **En Render, crea un nuevo servicio de tipo "Web Service".**
3. **Conecta tu repositorio de GitHub.**
4. **Configura los siguientes parámetros:**
   - **Build Command:**  
     ```
     pip install -r requirements.txt
     ```
   - **Start Command:**  
     ```
     gunicorn app:server
     ```
   - **Python Version:**  
     Render detecta automáticamente la versión desde tu `requirements.txt`, pero puedes especificar un archivo `runtime.txt` con el contenido `python-3.10.0` si lo deseas.
5. **Asegúrate de subir los archivos del modelo (`modelo_obesidad.pkl`, `target_encoder.pkl`, `encoders_dict.pkl`) al repositorio o súbelos manualmente a Render después del primer deploy.**
6. **Render expondrá automáticamente tu dashboard en una URL pública.**

## Estructura del Proyecto

```
parcial_metricas_de_rendimiento/
├── app.py                # Aplicación principal Dash
├── requirements.txt      # Dependencias del proyecto
├── Procfile              # Configuración para despliegue (Heroku/Gunicorn/Render)
├── .gitignore            # Archivos y carpetas ignorados por git
├── README.md             # Este archivo
├── modelo_obesidad.pkl   # (No incluido, agregar manualmente)
├── target_encoder.pkl    # (No incluido, agregar manualmente)
└── encoders_dict.pkl     # (No incluido, agregar manualmente)
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
