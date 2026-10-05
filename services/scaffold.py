from __future__ import annotations

from typing import Dict, List


def _build_webapp_files(project_name: str) -> Dict[str, str]:
    return {
        "package.json": """{
  \"name\": \"%s\",
  \"version\": \"1.0.0\",
  \"private\": true,
  \"scripts\": {
    \"dev\": \"vite\",
    \"build\": \"vite build\",
    \"preview\": \"vite preview\"
  },
  \"dependencies\": {
    \"react\": \"^18.3.1\",
    \"react-dom\": \"^18.3.1\"
  },
  \"devDependencies\": {
    \"vite\": \"^5.4.0\"
  }
}
""" % project_name,
        "index.html": """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>%s</title>
    <link rel="stylesheet" href="./styles.css" />
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="./main.js"></script>
  </body>
</html>
""" % project_name,
        "styles.css": """body {
  margin: 0;
  font-family: Arial, sans-serif;
  background: #0f172a;
  color: #e2e8f0;
}

#app {
  min-height: 100vh;
  display: grid;
  place-items: center;
}

.card {
  padding: 2rem;
  border-radius: 16px;
  background: rgba(15, 23, 42, 0.8);
  border: 1px solid rgba(148, 163, 184, 0.3);
}
""",
        "main.js": """const app = document.getElementById('app');
app.innerHTML = `
  <div class="card">
    <h1>%s</h1>
    <p>Built by N.O.R.M.A.L / Telvin</p>
  </div>
`;
""" % project_name,
    }


def _build_backend_files(project_name: str) -> Dict[str, str]:
    return {
        "requirements.txt": """fastapi\nuvicorn[standard]\npydantic\n""",
        "app/main.py": """from fastapi import FastAPI\n\napp = FastAPI(title=\"%s\")\n\n@app.get(\"/health\")\ndef health():\n    return {\"status\": \"ok\"}\n""" % project_name,
        "app/api/routes/chat.py": """from fastapi import APIRouter\n\nrouter = APIRouter()\n\n@router.get(\"/demo\")\ndef demo():\n    return {\"message\": \"Welcome to %s\"}\n""" % project_name,
        "app/services/brain.py": """def build_answer(prompt: str):\n    return f\"N.O.R.M.A.L response for: {prompt}\"\n""",
        "tests/test_health.py": """def test_health():\n    assert True\n""",
    }


def build_project_scaffold(project_name: str, app_type: str, description: str, tier: str = "OMNI") -> Dict:
    """Generate a structured app/web scaffold plan for Telvin / Appgrade workflows.

    Returns a plan and a starter file bundle that can be surfaced to the API consumer.
    """
    app_type = (app_type or "webapp").lower()
    tier = (tier or "OMNI").upper()

    if app_type in {"webapp", "website", "frontend"}:
        files = _build_webapp_files(project_name)
        project_files = [
            "package.json",
            "index.html",
            "styles.css",
            "main.js",
            "tests/test_smoke.py",
        ]
    elif app_type in {"api", "service", "backend"}:
        files = _build_backend_files(project_name)
        project_files = [
            "requirements.txt",
            "app/main.py",
            "app/api/routes/chat.py",
            "app/services/brain.py",
            "tests/test_health.py",
        ]
    else:
        files = _build_webapp_files(project_name)
        project_files = [
            "README.md",
            "src/app.py",
            "tests/test_app.py",
        ]

    phases = [
        "1. Requirements capture and user goals",
        "2. Architecture and folder structure",
        "3. Core data/model definitions",
        "4. API or UI wiring",
        "5. Validation, tests, and smoke run",
    ]

    validation = [
        "Verify dependencies and package manifests are correct.",
        "Run linting or syntax checks for generated files.",
        "Run smoke tests and confirm the app boots successfully.",
        "Check that environment variables and model paths are valid.",
        "Confirm the output matches the project specification.",
    ]

    return {
        "project_name": project_name,
        "app_type": app_type,
        "tier": tier,
        "description": description,
        "project_files": project_files,
        "files": [{"path": path, "content": content} for path, content in files.items()],
        "phases": phases,
        "validation": validation,
        "launch_steps": [
            "Install dependencies",
            "Set environment variables",
            "Run the local app or API server",
            "Execute smoke tests",
            "Review generated output and iterate on feedback",
        ],
    }
