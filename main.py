from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import subprocess

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="static")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    model: str = Form(...),
    prompt: str = Form(...),
    tokens: int = Form(...),
    temperature: float = Form(...)
):
    try:
        command = [
            'python', 'run_inference.py',
            '-m', f'models/{model}/ggml-model-i2_s.gguf',
            '-p', prompt,
            '-n', str(tokens),
            '-temp', str(temperature)
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = f"Error: {e.stderr}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prompt": prompt,
        "output": output
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
