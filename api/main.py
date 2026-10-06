# FastAPI entry point

from fastapi import FastAPI
from app.api.routes import chat, health, memory
from app.api.routes import generation

app = FastAPI(
    title="Innoxation AI Core",
    version="1.1.0"
)

app.include_router(chat.router)
app.include_router(memory.router)
app.include_router(health.router)
app.include_router(generation.router)