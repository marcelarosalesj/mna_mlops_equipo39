from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd

with open("app/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("app/model_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)


app = FastAPI()
column_names = [
    "Student Age",
    "Sex",
    "Graduated High-school Type",
    "Scholarship Type",
    "Additional Work",
    "Regular Artistic/Sports Activity",
    "Do you have a Partner",
    "Total Salary",
    "Transportation",
    "Accommodation in Cyprus",
    "Mothers Education",
    "Fathers Education",
    "Number of Siblings",
    "Parental Status",
    "Mothers Occupation",
    "Fathers Occupation",
    "Weekly Study Hours",
    "Reading Frequency (Non-Scientific)",
    "Reading Frequency (Scientific)",
    "Attendance to Seminars",
    "Impact on Success",
    "Attendance to Classes",
    "Preparation to Midterm 1",
    "Preparation to Midterm 2",
    "Taking Notes in Classes",
    "Listening in Classes",
    "Discussion Improves Success",
    "Flip-Classroom",
    "Cumulative GPA Last Semester",
    "Expected GPA at Graduation",
    "COURSE ID",
    # "OUTPUT Grade",
]


class PredictionRequest(BaseModel):
    data: dict[str, str] = Field(
        ...,
        min_items=31,
        max_items=31,
        description="Un diccionario con 32 claves de nombre específico y valores string que representan enteros",
    )


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Extraer y convertir los valores en el orden definido por column_names
        # data = np.array([[float(request.data[col]) for col in column_names]])
        df = pd.DataFrame([{col: float(request.data[col]) for col in column_names}])

        # Aplicar la transformación de característicaspipeline
        data = pipeline.transform(df)

        # Generar la predicción
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
