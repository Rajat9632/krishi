import os
import json
import google.generativeai as genai
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)
            user_prompt = data.get("user_prompt", "").strip()

            if not user_prompt:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No prompt provided"}).encode('utf-8'))
                return

            # This uses the Environment Variable you set in Vercel
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            genai.configure(api_key=api_key)
            ai_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            
            full_prompt = (
                f"You are a helpful agricultural AI assistant based in India. "
                f"Provide a helpful and informative response to the following query. "
                f"Keep your response concise but informative. Query: {user_prompt}"
            )

            response = ai_model.generate_content(full_prompt)
            ai_response = response.text.strip()

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"response": ai_response}).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        return