from fastapi import FastAPI

app = FastAPI(
    title="HWH-2-00am Incident Response AI",
    version="1.0.0",
    description="AI-powered incident response backend"
)


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "HWH-2-00am backend is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }