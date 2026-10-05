# /chat endpoint

import os

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.router import route_message
from app.services.scaffold import build_project_scaffold

router = APIRouter()


class ChatRequest(BaseModel):
    persona: str = "telvin"
    message: str


class ChatResponse(BaseModel):
    persona: str
    response: str


class AppScaffoldRequest(BaseModel):
    persona: str = "telvin"
    project_name: str
    app_type: str = "webapp"
    description: str
    tier: str = "OMNI"


class AppScaffoldResponse(BaseModel):
    persona: str
    tier: str
    project_name: str
    scaffold: str
    validation: str
    files: list[dict] = []


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    response = route_message(payload.persona, payload.message)
    return {"persona": payload.persona, "response": response}


@router.post("/scaffold", response_model=AppScaffoldResponse)
def scaffold_app(payload: AppScaffoldRequest):
    persona = payload.persona.lower()
    tier = payload.tier.upper()
    project_name = payload.project_name.strip() or "normal-app"

    scaffold_data = build_project_scaffold(
        project_name=project_name,
        app_type=payload.app_type,
        description=payload.description,
        tier=tier,
    )

    scaffold_text = (
        f"Project: {scaffold_data['project_name']}\n"
        f"Type: {scaffold_data['app_type']}\n"
        f"Tier: {scaffold_data['tier']}\n\n"
        f"Plan:\n- " + "\n- ".join(scaffold_data["phases"]) + "\n\n"
        f"Key files:\n- " + "\n- ".join(scaffold_data["project_files"]) + "\n\n"
        f"Validation:\n- " + "\n- ".join(scaffold_data["validation"]) + "\n\n"
        f"Launch:\n- " + "\n- ".join(scaffold_data["launch_steps"])
    )

    scaffold = route_message(persona, scaffold_text)
    validation = "Validation steps: confirm folder structure, verify package manifests, ensure dependencies match the app type, check build scripts, run a smoke test, and validate generated code compiles."
    return {
        "persona": persona,
        "tier": tier,
        "project_name": project_name,
        "scaffold": scaffold,
        "validation": validation,
        "files": scaffold_data["files"],
    }


class ProjectWriteRequest(AppScaffoldRequest):
    pass


class ProjectWriteResponse(BaseModel):
    persona: str
    tier: str
    project_name: str
    output_dir: str
    created_files: list[str]


@router.post("/scaffold/write", response_model=ProjectWriteResponse)
def write_scaffold(payload: ProjectWriteRequest):
    persona = payload.persona.lower()
    tier = payload.tier.upper()
    project_name = payload.project_name.strip() or "normal-app"
    project_dir = os.path.join("data", "projects", project_name)
    os.makedirs(project_dir, exist_ok=True)

    scaffold_data = build_project_scaffold(
        project_name=project_name,
        app_type=payload.app_type,
        description=payload.description,
        tier=tier,
    )

    created_files: list[str] = []
    for file_item in scaffold_data["files"]:
        rel_path = file_item["path"]
        full_path = os.path.join(project_dir, *rel_path.split("/"))
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as handle:
            handle.write(file_item["content"])
        created_files.append(rel_path)

    return {
        "persona": persona,
        "tier": tier,
        "project_name": project_name,
        "output_dir": project_dir,
        "created_files": created_files,
    }


