"""External service connectors used by generation pipelines.

Provide a simple `ExternalConnector` class that can be extended to support
APIs like search services, knowledge bases, or other web endpoints.
"""

from __future__ import annotations

import requests
from typing import Dict, Optional


class ExternalConnector:
    def __init__(self, name: str, base_url: str, headers: Optional[Dict[str, str]] = None):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}

    def fetch(self, path: str = "", params: Optional[Dict] = None, method: str = "GET", json_body: Optional[Dict] = None, timeout: int = 10):
        if path:
            url = f"{self.base_url}/{path.lstrip('/')}"
        else:
            url = self.base_url
        try:
            if method.upper() == "GET":
                resp = requests.get(url, params=params, headers=self.headers, timeout=timeout)
            else:
                resp = requests.post(url, json=json_body, params=params, headers=self.headers, timeout=timeout)
            resp.raise_for_status()
            # try json else text
            try:
                return resp.json()
            except Exception:
                return resp.text
        except Exception as e:
            return {"error": str(e)}
