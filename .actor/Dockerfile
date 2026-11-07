FROM apify/actor-python:3.11

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# DIAGNÓSTICO (paso 1): solo imprime versiones, nunca falla el build.
RUN python --version; pip --version; pip list 2>&1 | grep -iE "apify|crawlee|pydantic|scikit-learn|numpy|pandas" || echo "(grep sin coincidencias)"

# DIAGNÓSTICO (paso 2): intenta importar el SDK y muestra el traceback completo si falla.
RUN python -c "import traceback
try:
    from apify import Actor
    print('apify import OK')
except Exception:
    traceback.print_exc()
    raise"

COPY . ./

RUN python -c "import joblib; m = joblib.load('rf_modelo-final.joblib'); print('modelo OK:', type(m).__name__, m.n_features_in_)"

CMD ["python", "-m", "main"]
