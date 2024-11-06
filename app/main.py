from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conlist
import pickle
import numpy as np

with open("app/model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()


class PredictionRequest(BaseModel):
    data: dict[str, str] = Field(
        ...,
        min_items=62,
        max_items=62,
        description="Un diccionario con 62 claves y valores de tipo string representando flotantes",
    )


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        data = np.array([[float(request.data[str(i)]) for i in range(62)]])
        prediction = model.predict(data)[0]
        return {"prediction": prediction}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Falta la clave: {str(e)}")
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Error en formato de datos: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
