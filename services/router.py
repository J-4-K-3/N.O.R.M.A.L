# chooses persona + logic

from app.core.personas import PERSONAS
from app.services.brain import generate_response


def route_message(persona_key, message):
    persona = PERSONAS.get(persona_key, PERSONAS["telvin"])

    # pass persona_key so backends may choose different tiers or defaults
    return generate_response(message, persona, persona_key)