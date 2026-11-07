# Actor Apify: score FICO desde CSV

## Qué hace
- Recibe un CSV mediante el campo `inputFile`.
- Carga `rf_modelo-final.joblib`.
- Calcula `fico_score` para cada fila válida.
- Agrega `error` a la salida. Si una fila tiene datos faltantes o no numéricos, registra el error y deja `fico_score` vacío para esa fila.
- Continúa procesando las demás filas.
- Guarda el CSV completo en el Key-Value Store con la clave `OUTPUT`.
- Publica los registros en el Dataset de la ejecución y un resumen en `SUMMARY`.

## Columnas requeridas
El modelo utiliza estas cinco variables, en este orden:
1. `bc_open_to_buy`
2. `revol_util`
3. `is_verified`
4. `pct_tl_nvr_dlq`
5. `pub_rec_bankruptcies`

También acepta `verification_status` en lugar de `is_verified`. En ese caso aplica la regla del entrenamiento:
- `Verified` o `Source Verified` -> `1`
- cualquier otro valor no vacío -> `0`

`is_verified` debe ser 0 o 1.

## Cómo desplegar
1. En Apify Console, crea un Actor nuevo desde código fuente.
2. Sube el contenido de esta carpeta (o importa el ZIP).
3. Construye el Actor.
4. En la pestaña Input, carga el CSV en `CSV de entrada`.
5. Ejecuta el Actor.
6. Abre el Key-Value Store de la ejecución y descarga `OUTPUT` (CSV). También puedes revisar el Dataset y `SUMMARY`.

## Formato de salida
Se conservan todas las columnas originales y se agregan:
- `fico_score`: predicción del Random Forest, redondeada al entero más cercano.
- `error`: vacío si la fila se calculó; descripción del problema si no se calculó.

Una fila inválida no detiene el procesamiento del resto del archivo.

## Nota sobre el modelo
El script de entrenamiento define `fico_prom` como el promedio de `fico_range_low` y `fico_range_high`; por lo tanto, el actor devuelve una **estimación del valor medio FICO**, no un puntaje FICO oficial ni una decisión crediticia. La salida es una predicción del modelo.
