import io
import logging
import os
import requests
import pandas as pd
import joblib

from apify import Actor

MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_modelo-final.joblib")
FEATURES = [
    "bc_open_to_buy",
    "revol_util",
    "is_verified",
    "pct_tl_nvr_dlq",
    "pub_rec_bankruptcies",
]

def normalize_verification_status(value):
    if pd.isna(value):
        return None
    return 1 if str(value).strip() in ("Verified", "Source Verified") else 0

async def main():
    async with Actor:
        actor_input = await Actor.get_input() or {}
        file_url = actor_input.get("inputFile")

        if not file_url:
            raise ValueError(
                "Falta 'inputFile'. Carga un CSV en el campo de archivo de entrada."
            )

        response = requests.get(file_url, timeout=120)
        response.raise_for_status()

        try:
            df = pd.read_csv(io.BytesIO(response.content))
        except Exception as exc:
            raise ValueError(f"No se pudo leer el CSV: {exc}") from exc

        if df.empty:
            raise ValueError("El CSV no contiene clientes.")

        # Acepta verification_status como alternativa a is_verified,
        # usando exactamente la regla aplicada durante el entrenamiento.
        if "is_verified" not in df.columns and "verification_status" in df.columns:
            df["is_verified"] = df["verification_status"].map(
                normalize_verification_status
            )

        model = joblib.load(MODEL_PATH)

        # Verifica que el artefacto corresponda a las cinco variables esperadas.
        model_features = list(getattr(model, "feature_names_in_", []))
        if model_features and model_features != FEATURES:
            raise ValueError(
                "Las variables del modelo no coinciden con las esperadas. "
                f"Modelo: {model_features}; esperadas: {FEATURES}"
            )

        # Inicializa las columnas de salida conservando todos los registros.
        df["fico_score"] = pd.NA
        df["error"] = ""

        valid_rows = []
        valid_features = []

        for idx, row in df.iterrows():
            missing = [
                col for col in FEATURES
                if col not in df.columns or pd.isna(row.get(col))
            ]

            if missing:
                message = "Datos faltantes: " + ", ".join(missing)
                df.at[idx, "error"] = message
                Actor.log.error("Fila %s: %s", idx + 1, message)
                continue

            values = {}
            invalid = []
            for col in FEATURES:
                try:
                    values[col] = float(row[col])
                except (TypeError, ValueError):
                    invalid.append(col)

            if invalid:
                message = "Datos no numéricos: " + ", ".join(invalid)
                df.at[idx, "error"] = message
                Actor.log.error("Fila %s: %s", idx + 1, message)
                continue

            # is_verified debe ser binaria.
            if values["is_verified"] not in (0.0, 1.0):
                message = "is_verified debe ser 0 o 1"
                df.at[idx, "error"] = message
                Actor.log.error("Fila %s: %s", idx + 1, message)
                continue

            valid_rows.append(idx)
            valid_features.append(values)

        if valid_rows:
            X = pd.DataFrame(valid_features, columns=FEATURES)
            predictions = model.predict(X)
            for idx, prediction in zip(valid_rows, predictions):
                # La salida se redondea al entero más cercano.
                df.at[idx, "fico_score"] = int(round(float(prediction)))

        # CSV final disponible en el Key-Value Store como OUTPUT.csv.
        csv_text = df.to_csv(index=False)
        await Actor.set_value(
            "OUTPUT",
            csv_text,
            content_type="text/csv; charset=utf-8"
        )

        # También publica cada cliente en el dataset de Apify.
        for record in df.where(pd.notna(df), None).to_dict(orient="records"):
            await Actor.push_data(record)

        failed = int((df["error"] != "").sum())
        succeeded = len(df) - failed

        await Actor.set_value("SUMMARY", {
            "total_clientes": len(df),
            "calculados": succeeded,
            "con_error": failed,
            "archivo_csv": "OUTPUT"
        })

        Actor.log.info(
            "Proceso terminado. Clientes: %s | Calculados: %s | Con error: %s",
            len(df), succeeded, failed
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
