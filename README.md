# MercadoFútbol

Aplicación académica en Python para estimar el valor de mercado de jugadores de fútbol usando regresión lineal múltiple. El proyecto adapta los requisitos del documento de Álgebra Lineal a un escenario de mercado de fichajes.

## Funcionalidades

- Inicio de sesión visual con usuario `admin` y contraseña `futbol2025`.
- Carga y validación de archivos CSV.
- Tabla visual de jugadores y datos usados en el entrenamiento.
- Modelo de regresión lineal múltiple calculado con la ecuación normal.
- Predicción del valor de un nuevo jugador.
- Métricas MAE, RMSE y R².
- Gráficas embebidas en la interfaz: real vs. estimado, valoración vs. precio y ranking de jugadores.
- Guardado de la gráfica actual como PNG, PDF o SVG.
- Al calcular una predicción, se genera una gráfica comparativa visible en la sección **Gráficas** y se guarda automáticamente como PNG en `outputs/graficas_predicciones/`.

## Variables del modelo

| Variable | Descripción |
| --- | --- |
| `Edad` | Edad del jugador |
| `Valoracion` | Valoración general de 0 a 100 |
| `Potencial` | Proyección deportiva de 0 a 100 |
| `Goles_Asistencias` | Producción ofensiva combinada |
| `Valor_Mercado_M` | Valor objetivo en millones de euros |

La matriz de diseño incluye una columna de unos para el intercepto. Los coeficientes se calculan mediante `β = (XᵀX)⁻¹XᵀY`; el código utiliza la pseudoinversa de Moore-Penrose para mantener estabilidad numérica.

## Ejecución

```bash
python -m pip install -r requirements.txt
python app.py
```

El CSV de ejemplo está en `data/jugadores.csv`. Para cargar otro archivo, conserva las mismas columnas:

```text
Jugador,Edad,Valoracion,Potencial,Goles_Asistencias,Valor_Mercado_M
```

Los datos incluidos son ficticios y se utilizan únicamente con fines académicos.
