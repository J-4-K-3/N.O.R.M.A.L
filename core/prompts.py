# system prompts builder

from typing import Dict, Optional

from app.core.config import ENGINE_IDENTITY, resolve_tier_config


def build_persona_prompt(persona: Dict, tier: Optional[str] = None) -> str:
    """Create a behavior-first persona prompt.

    The runtime identity is defined in engine config; the persona prompt handles
    behavior, style and role without hard-coding company claims in every persona.
    """
    name = persona.get("name", "AI")
    style = persona.get("style", "helpful")
    tier_key = (tier or "EB").upper()
    tier_config = resolve_tier_config(tier_key)

    tier_label = tier_config["label"]
    tier_name = tier_config["name"]
    tier_behavior = tier_config["system_behavior"]
    tools = ", ".join(tier_config.get("tools", []))
    capabilities = ", ".join(tier_config.get("capabilities", []))

    identity_block = (
        f"Engine: {ENGINE_IDENTITY['name']} ({ENGINE_IDENTITY['full_name']})\n"
        f"Organization: {ENGINE_IDENTITY['organization']}\n"
        f"CEO: {ENGINE_IDENTITY['ceo']}\n"
        f"Version: {ENGINE_IDENTITY['version']}\n"
    )

    persona_block = (
        f"Persona: {name}\n"
        f"Style: {style}\n"
        f"Behavior: {tier_behavior}\n"
    )

    tier_block = (
        f"Tier: {tier_label} ({tier_name})\n"
        f"Available tools: {tools}\n"
        f"Capabilities: {capabilities}\n"
    )

    rules = (
        "Rules:\n"
        "- Be truthful about what the system is: N.O.R.M.A.L is the engine runtime operated by Innoxation Tech Inc.\n"
        "- Preserve the active persona and respond in that voice, but keep the system identity separate from the persona.\n"
        "- If asked about provenance, say you are part of N.O.R.M.A.L / Innoxation Tech Inc.\n"
        "- For coding or app-building tasks, plan before writing code and prefer clean, validated output.\n"
        "- If the task is uncertain or under-specified, ask one precise clarifying question.\n"
    )

    return (
        f"{identity_block}\n"
        f"{tier_block}\n"
        f"{persona_block}\n"
        f"{rules}\n"
    )

