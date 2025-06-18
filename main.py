from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from utils.preprocess import preprocess_image
from models.load_models import tomato_model, tomato_img_size, banana_model, banana_img_size

app = FastAPI(
    title="Harvest Readiness Classifier",
    description="API for classifying if tomatoes or bananas are ready for harvest"
)

@app.post("/predict/tomato")
async def predict_tomato(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_image = preprocess_image(contents, tomato_img_size)
        prediction = tomato_model.predict(processed_image)[0][0]
        ready = bool(prediction > 0.5)
        return JSONResponse(content={
            "ready_for_harvest": ready,
            "crop": "tomato",
            "confidence": float(prediction),
            "model_input_size": tomato_img_size
        })
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/predict/banana")
async def predict_banana(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_image = preprocess_image(contents, banana_img_size)
        prediction = banana_model.predict(processed_image)[0][0]
        ready = bool(prediction > 0.5)
        return JSONResponse(content={
            "ready_for_harvest": ready,
            "crop": "banana",
            "confidence": float(prediction),
            "model_input_size": banana_img_size
        })
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/")
async def root():
    return {
        "message": "Harvest Readiness Classifier API",
        "endpoints": {
            "tomato": "/predict/tomato",
            "banana": "/predict/banana"
        }
    }
