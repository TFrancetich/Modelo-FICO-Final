FROM apify/actor-python:3.11

# La imagen base NO incluye el SDK de Apify: se instala desde requirements.txt.
# apify 3.x trae su propio crawlee compatible con pydantic.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Muestra versiones instaladas (nunca falla el build).
RUN python --version; pip list 2>&1 | grep -iE "apify|crawlee|pydantic|scikit-learn|numpy|pandas" || echo "(sin coincidencias)"

# Falla en el BUILD (con traceback completo) si el SDK no se puede importar.
RUN python -c "from apify import Actor; print('apify import OK')"

COPY . ./

RUN python -c "import joblib; m = joblib.load('rf_modelo-final.joblib'); print('modelo OK:', type(m).__name__, m.n_features_in_)"

CMD ["python", "-m", "main"]
