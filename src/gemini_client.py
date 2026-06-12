# src/gemini_client.py

import os

from google import genai


class GeminiClient:

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found."
            )

        self.client = genai.Client(
            api_key=api_key
        )

    def generate_response(
        self,
        ticket_text: str,
        predicted_queue: str
    ) -> str:

        prompt = f"""
You are a professional customer support assistant.

Customer Ticket:
{ticket_text}

Predicted Queue:
{predicted_queue}

Generate a polite acknowledgement response.

Requirements:
- Thank the customer.
- Acknowledge the issue.
- Mention the ticket was routed to the {predicted_queue} team.
- Do not promise resolution times.
- Keep response under 120 words.
- Professional and empathetic tone.
- Return only the response text.
"""

        try:

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            return response.text.strip()

        except Exception as e:

            print("Gemini Error:", e)

            return f"""
Dear Customer,

Thank you for contacting us.

We have received your request and routed it to our {predicted_queue} team for review.

Our support team is currently evaluating your concern and will assist you as soon as possible.

We appreciate your patience.

Kind regards,
Customer Support Team
""".strip()