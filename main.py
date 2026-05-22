"""
Deriv AI Assessment — Gemini API Skeleton
==========================================
Minimal setup to verify the Gemini API connection works before the assessment.
Extend this file during the assessment to implement the required functionality.

Usage:
    python main.py
"""

import json
import os
import sys

import google.genai as genai
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Gemini API Client
# ---------------------------------------------------------------------------

def get_client() -> genai.Client:
    """Initialise and return a Gemini API client."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_key_here":
        print("ERROR: GOOGLE_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)
    return genai.Client(api_key=GOOGLE_API_KEY)


# ---------------------------------------------------------------------------
# Core Function — extend this during the assessment
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, response_format: str = "text") -> str | dict:
    """Send a prompt to Gemini and return the response.

    Args:
        prompt: The text prompt to send.
        response_format: "text" for plain text, "json" for structured JSON.

    Returns:
        The response as a string or parsed dict/list.
    """
    client = get_client()

    config = {
        "temperature": 0.1,
    }
    if response_format == "json":
        config["response_mime_type"] = "application/json"

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config,
        )
    except Exception as e:
        print(f"ERROR: Gemini API call failed — {e}", file=sys.stderr)
        sys.exit(1)

    if response_format == "json":
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            print("ERROR: Expected JSON but got invalid response", file=sys.stderr)
            sys.exit(1)

    return response.text


# ---------------------------------------------------------------------------
# Main — verification test
# ---------------------------------------------------------------------------

def main():
    """Verify the Gemini API connection works."""
    print("Testing Gemini API connection...")

    result = call_gemini("Reply with exactly: CONNECTION_OK")
    print(f"Response: {result.strip()}")

    if "CONNECTION_OK" in result:
        print("SUCCESS: Gemini API is working.")
    else:
        print("WARNING: Unexpected response — check your setup.")


if __name__ == "__main__":
    main()
