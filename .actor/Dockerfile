FROM apify/actor-python:3.11

# apify/crawlee/pydantic ya vienen en la imagen base como un conjunto compatible.
# No se listan en requirements.txt para que pip no los reinstale con versiones desparejas.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Falla en el BUILD (no en la ejecución) si el SDK de Apify o el modelo no cargan.
RUN pip list 2>/dev/null | grep -iE "^(apify|crawlee|pydantic|scikit-learn|numpy|pandas) " \
    && python -c "from apify import Actor; print('apify import OK')"

COPY . ./

RUN python -c "import joblib; m = joblib.load('rf_modelo-final.joblib'); print('modelo OK:', type(m).__name__, m.n_features_in_)"

CMD ["python", "-m", "main"]
