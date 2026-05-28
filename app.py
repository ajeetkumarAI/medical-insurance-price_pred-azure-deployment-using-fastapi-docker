import json
import pickle

import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Load model and metadata once at startup
with open("Linear_Model.pkl", "rb") as f:
    model = pickle.load(f)

with open("project_data.json", "r") as f:
    project_data = json.load(f)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(["html", "xml"]),
)


def render_template(name: str, **context) -> HTMLResponse:
    template = env.get_template(name)
    return HTMLResponse(template.render(**context))

VALID_SEX = {"male", "female"}
VALID_SMOKER = {"yes", "no"}
VALID_REGION = {"southwest", "southeast", "northwest", "northeast"}


def predict_insurance(age, sex, bmi, children, smoker, region):
    test_array = np.zeros(model.n_features_in_)
    test_array[0] = age
    test_array[1] = project_data["sex"][sex]
    test_array[2] = bmi
    test_array[3] = children
    test_array[4] = project_data["smoker"][smoker]
    region_col = f"region_{region}"
    region_index = project_data["columns"].index(region_col)
    test_array[region_index] = 1
    return round(model.predict([test_array])[0], 2)


@app.get("/")
async def index(request: Request):
    return render_template("index.html")


@app.post("/predict")
async def predict(
    request: Request,
    age: float = Form(...),
    sex: str = Form(...),
    bmi: float = Form(...),
    children: float = Form(...),
    smoker: str = Form(...),
    region: str = Form(...),
):
    try:
        sex = sex.strip().lower()
        smoker = smoker.strip().lower()
        region = region.strip().lower()

        if sex not in VALID_SEX:
            return render_template("index.html", error="Invalid sex value.")
        if smoker not in VALID_SMOKER:
            return render_template("index.html", error="Invalid smoker value.")
        if region not in VALID_REGION:
            return render_template("index.html", error="Invalid region value.")
        if age < 0 or age > 150:
            return render_template("index.html", error="Age must be between 0 and 150.")
        if bmi < 0:
            return render_template("index.html", error="BMI must be non-negative.")
        if children < 0:
            return render_template("index.html", error="Children must be non-negative.")

        charges = predict_insurance(age, sex, bmi, children, smoker, region)
        return render_template("index.html", charges=charges)

    except (KeyError, ValueError) as e:
        return render_template("index.html", error=f"Invalid input: {e}")


@app.get("/health")
async def health():
    return JSONResponse({"status": "healthy"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
