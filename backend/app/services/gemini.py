import os
from pathlib import Path
from google import genai
from dotenv import load_dotenv

# Load .env from backend directory
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class GeminiService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        self.client = genai.Client(api_key=api_key)

    def generate_reply(self, ticket_text, predicted_queue, ticket_number, client_name):
        prompt = f"""
You are Shanyan AI Bot, a customer support assistant for Shanyan AI company.

Ticket Number: {ticket_number}
Ticket Category: {predicted_queue}
Customer Name: {client_name}
Customer Message: {ticket_text}

Write a professional acknowledgement reply with this EXACT format:

Subject: Regarding your recent {predicted_queue.lower()} issue - [Ticket Number - {ticket_number}]

Dear {client_name},

Thank you for reaching out to us. We understand your concern regarding [summarize their issue briefly].

We sincerely apologize for any inconvenience this may cause. Please be assured that we are reviewing this issue and will investigate it immediately. We will be in touch with an update as soon as possible.

Sincerely,
Shanyan AI Bot
Shanyan AI Customer Support

Do NOT add any additional text outside this format.
"""
        response = self.client.models.generate_content(
            model='gemma-3-27b-it',
            contents=prompt
        )
        return response.text.strip()
