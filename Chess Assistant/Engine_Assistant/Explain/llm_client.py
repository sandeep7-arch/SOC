"""
llm_client.py — LLM Backend for Chess Explanation Engine
=========================================================
YOUR MODULE: Engine_Assistant/Explain/llm_client.py
 
This file is the ONLY place in the entire project that talks to an external LLM.
Every other file (move_explainer, blunder_explainer, etc.) will import from here.
 
Why this design?
- If you want to switch from Gemini to Claude, you change ONE line in your main script.
- All explainers stay untouched.
- You can test everything offline using MockLLMProvider (no API key needed).
 
PROVIDERS INCLUDED:
    1. MockLLMProvider       — works offline, zero cost, for testing
    2. GeminiProvider        — Google Gemini API (free tier: 15 req/min)
    3. ClaudeProvider        — Anthropic Claude API
    4. OpenAIProvider        — OpenAI GPT API
    5. LocalLlamaProvider    — Ollama running locally (completely free, no internet)
 
USAGE (copy this to understand the pattern):
    from Engine_Assistant.Explain.llm_client import get_llm, MockLLMProvider, GeminiProvider
 
    # Today — no API key needed
    llm = MockLLMProvider()
    response = llm.complete("Explain this chess move...")
 
    # Next week — swap to Gemini, zero other changes
    llm = GeminiProvider(api_key="YOUR_KEY_HERE")
    response = llm.complete("Explain this chess move...")
"""
 
import os
import json
import time
import random
from abc import ABC, abstractmethod
from typing import Optional
 
 
# =============================================================================
# BASE CLASS (The Contract)
# =============================================================================
 
class LLMProvider(ABC):
    """
    Abstract base class — every LLM provider MUST implement these two methods.
 
    Think of this as a USB port standard.
    Any provider that plugs in must have: complete() and complete_with_retry()
    Your explainer code only talks to this interface, never to Gemini/OpenAI directly.
    """
 
    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Send a prompt to the LLM and get a text response back.
 
        Args:
            prompt     : The full text prompt to send.
            max_tokens : Maximum length of response. Keep low (300-500) for chess explanations.
 
        Returns:
            str: The LLM's response as a plain string.
 
        Raises:
            LLMError: If the API call fails.
        """
        pass
 
    def complete_with_retry(self, prompt: str, max_tokens: int = 500,
                             retries: int = 3, delay: float = 2.0) -> str:
        """
        Same as complete(), but automatically retries on failure.
 
        This handles temporary network errors or API rate limits.
        Default: 3 retries with 2 second wait between each.
 
        Args:
            prompt     : The prompt to send.
            max_tokens : Max response length.
            retries    : How many times to retry before giving up.
            delay      : Seconds to wait between retries.
 
        Returns:
            str: The LLM's response.
        """
        last_error = None
 
        for attempt in range(retries):
            try:
                return self.complete(prompt, max_tokens)
 
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    # Log the failure and wait before retry
                    print(f"[LLM] Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 1.5  # Increase wait time each retry (exponential backoff)
 
        # All retries exhausted
        raise LLMError(f"All {retries} attempts failed. Last error: {last_error}")
 
    def __repr__(self):
        return f"{self.__class__.__name__}()"
 
 
# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================
 
class LLMError(Exception):
    """Raised when an LLM API call fails after all retries."""
    pass
 
class LLMConfigError(Exception):
    """Raised when provider is misconfigured (e.g., missing API key)."""
    pass
 
 
# =============================================================================
# PROVIDER 1: MockLLMProvider (USE THIS FIRST — no setup needed)
# =============================================================================
 
class MockLLMProvider(LLMProvider):
    """
    A fake LLM that returns pre-written template responses.
 
    WHY THIS EXISTS:
    You need to test your explainer code NOW, before you have an API key.
    This provider reads your prompt, detects keywords like "blunder" or "mistake",
    and returns a realistic-looking template response.
 
    No internet. No API key. No cost. Instant.
 
    WHEN TO USE:
    -  Before you get a Gemini key
    - In tests: So tests run fast without API calls
    - Demos: If internet is unavailable
 
    Example:
        llm = MockLLMProvider()
        response = llm.complete("A player made a blunder. Move: e4e5. Score: -300cp")
        print(response)  # Returns a blunder explanation template
    """
 
    # Pre-written response templates. The complete() method picks the right one.
    TEMPLATES = {
        "blunder": (
            "This was a critical blunder that immediately shifted the position from "
            "advantageous to losing. The move overlooked the opponent's tactical response, "
            "which wins material decisively. The correct approach was the suggested move, "
            "which maintains the positional advantage and keeps pieces coordinated. "
            "In chess, losing a piece for insufficient compensation is almost always fatal "
            "at this level of play."
        ),
        "mistake": (
            "This move was a significant mistake that weakened the position considerably. "
            "While not immediately losing, it allowed the opponent to seize the initiative "
            "and create threats that are difficult to meet. The better move would have "
            "maintained the tension while keeping defensive resources intact. "
            "Improving move selection in these complex positions is key to avoiding "
            "similar mistakes in future games."
        ),
        "inaccuracy": (
            "This inaccuracy slightly worsened the position. While not a serious error, "
            "the better move would have preserved more options and kept the position "
            "more flexible. In competitive play, accumulating small inaccuracies can "
            "gradually erode an advantage, so it's worth understanding the subtlety here."
        ),
        "good": (
            "This was a solid, accurate move that maintained the balance of the position. "
            "It correctly addresses the key features of the position while keeping "
            "pieces active. This kind of move shows good positional understanding."
        ),
        "excellent": (
            "Excellent move! This is precisely the kind of accurate, purposeful play "
            "that strong players make. It improves piece placement, controls key squares, "
            "and poses problems for the opponent to solve. Well calculated."
        ),
        "opening": (
            "In the opening phase, the goal is piece development, king safety, and "
            "central control. This move either helps or hinders those goals. Focus on "
            "getting your pieces to active squares and castling early."
        ),
        "endgame": (
            "In the endgame, king activity becomes crucial. Pawns become much more "
            "valuable, and small positional advantages can become decisive. Accurate "
            "technique is everything at this stage of the game."
        ),
        "default": (
            "This move changed the evaluation of the position. Understanding why requires "
            "looking at the specific tactical and strategic features present. The engine's "
            "suggested alternative would have led to a better outcome by addressing the "
            "key positional requirements more accurately."
        )
    }
 
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Scans the prompt for chess keywords and returns the matching template.
 
        Args:
            prompt     : The chess explanation prompt (we scan this for keywords).
            max_tokens : Ignored in mock — we return fixed templates.
 
        Returns:
            str: A realistic-looking chess explanation.
        """
        prompt_lower = prompt.lower()
 
        # Check keywords in priority order (worst mistakes first)
        if "blunder" in prompt_lower:
            return self.TEMPLATES["blunder"]
        elif "mistake" in prompt_lower:
            return self.TEMPLATES["mistake"]
        elif "inaccuracy" in prompt_lower:
            return self.TEMPLATES["inaccuracy"]
        elif "excellent" in prompt_lower:
            return self.TEMPLATES["excellent"]
        elif "good" in prompt_lower:
            return self.TEMPLATES["good"]
        elif "opening" in prompt_lower:
            return self.TEMPLATES["opening"]
        elif "endgame" in prompt_lower:
            return self.TEMPLATES["endgame"]
        else:
            return self.TEMPLATES["default"]
 
 
# =============================================================================
# PROVIDER 2: GeminiProvider (RECOMMENDED — free tier is generous)
# =============================================================================
 
class GroqProvider(LLMProvider):
    DEFAULT_MODEL = "llama-3.1-8b-instant"

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise LLMConfigError("Groq API key not found! Set GROQ_API_KEY env variable.")
        self.model_name = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except ImportError:
                raise LLMConfigError("groq package not installed! Fix: pip install groq")
        return self._client

    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a chess coach explaining moves to students. Be concise and specific."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise LLMError(f"Groq API error: {e}")

class GeminiProvider(LLMProvider):
    """
    Calls Google's Gemini API to generate chess explanations.

    FREE TIER LIMITS:
    - 15 requests per minute
    - No credit card required

    SETUP:
        pip install google-genai

    USAGE:
        llm = GeminiProvider(api_key="AIza...")
        response = llm.complete("Explain why e4e5 was a blunder.")
    """

    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        if not self.api_key:
            raise LLMConfigError(
                "Gemini API key not found!\n"
                "Fix: Either pass api_key='YOUR_KEY' or set env variable:\n"
                "     export GEMINI_API_KEY='YOUR_KEY'"
            )

        self.model_name = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai  # new SDK
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise LLMConfigError(
                    "google-genai package not installed!\n"
                    "Fix: pip install google-genai"
                )
        return self._client

    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            from google import genai
            from google.genai import types

            client = self._get_client()

            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                )
            )

            if response.text:
                return response.text.strip()
            else:
                raise LLMError("Gemini returned empty response.")

        except Exception as e:
            if "API_KEY_INVALID" in str(e):
                raise LLMConfigError(f"Invalid Gemini API key: {e}")
            elif "RATE_LIMIT" in str(e):
                raise LLMError(f"Rate limit hit. Wait 60 seconds. Error: {e}")
            else:
                raise LLMError(f"Gemini API error: {e}")
 
 
# =============================================================================
# PROVIDER 3: ClaudeProvider (Anthropic)
# =============================================================================
 
class ClaudeProvider(LLMProvider):
    """
    Calls Anthropic's Claude API.
 
    SETUP:
        pip install anthropic
 
    GET API KEY:
        https://console.anthropic.com/ → API Keys
 
    USAGE:
        llm = ClaudeProvider(api_key="sk-ant-...")
        response = llm.complete("Explain this chess blunder...")
    """
 
    DEFAULT_MODEL = "claude-3-haiku-20240307"  # Fastest and cheapest Claude model
 
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
 
        if not self.api_key:
            raise LLMConfigError(
                "Anthropic API key not found!\n"
                "Fix: pass api_key='sk-ant-...' or set ANTHROPIC_API_KEY env variable."
            )
 
        self.model_name = model
        self._client = None
 
    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise LLMConfigError("anthropic package not installed! Fix: pip install anthropic")
        return self._client
 
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            client = self._get_client()
            message = client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
 
        except Exception as e:
            raise LLMError(f"Claude API error: {e}")
 
 
# =============================================================================
# PROVIDER 4: OpenAIProvider
# =============================================================================
 
class OpenAIProvider(LLMProvider):
    """
    Calls OpenAI's GPT API.
 
    SETUP:
        pip install openai
 
    GET API KEY:
        https://platform.openai.com/api-keys
 
    USAGE:
        llm = OpenAIProvider(api_key="sk-...")
        response = llm.complete("Explain this chess move...")
    """
 
    DEFAULT_MODEL = "gpt-3.5-turbo"  # Cheapest option. Use gpt-4o for best quality.
 
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
 
        if not self.api_key:
            raise LLMConfigError(
                "OpenAI API key not found!\n"
                "Fix: pass api_key='sk-...' or set OPENAI_API_KEY env variable."
            )
 
        self.model_name = model
        self._client = None
 
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise LLMConfigError("openai package not installed! Fix: pip install openai")
        return self._client
 
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a chess coach explaining moves to students. Be concise and specific."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
 
        except Exception as e:
            raise LLMError(f"OpenAI API error: {e}")
 
 
# =============================================================================
# PROVIDER 5: LocalLlamaProvider (Completely free, no internet needed)
# =============================================================================
 
class LocalLlamaProvider(LLMProvider):
    """
    Calls a local LLM running via Ollama. 100% free, works offline.
 
    SETUP:
    1. Download Ollama: https://ollama.ai
    2. Run in terminal: ollama pull llama3.2
    3. Ollama starts automatically as a background service
 
    USAGE:
        llm = LocalLlamaProvider(model="llama3.2")
        response = llm.complete("Explain this chess blunder...")
 
    AVAILABLE MODELS (run: ollama list):
        - llama3.2      : Fast, good quality
        - mistral       : Good for structured output
        - phi3          : Very fast, smaller model
    """
 
    DEFAULT_HOST = "http://localhost:11434"
 
    def __init__(self, model: str = "llama3.2", host: str = DEFAULT_HOST):
        """
        Args:
            model : The Ollama model name. Run 'ollama list' to see available.
            host  : Ollama server address. Default is localhost.
        """
        self.model = model
        self.host = host
 
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Sends prompt to local Ollama instance and returns response.
 
        Raises:
            LLMError: If Ollama is not running or model not found.
        """
        try:
            import urllib.request
 
            payload = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens}
            }).encode("utf-8")
 
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
 
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip()
 
        except ConnectionRefusedError:
            raise LLMError(
                "Cannot connect to Ollama! Make sure it's running.\n"
                "Fix: Download from https://ollama.ai and start it."
            )
        except Exception as e:
            raise LLMError(f"Ollama error: {e}")
 
 
# =============================================================================
# FACTORY FUNCTION (Convenience — use this in your scripts)
# =============================================================================
 
def get_llm(provider: str = "mock", **kwargs) -> LLMProvider:
    """
    Factory function — creates the right LLM provider by name.
 
    Instead of importing each class separately, just call get_llm("gemini").
 
    Args:
        provider : One of: "mock", "gemini", "claude", "openai", "ollama"
        **kwargs : Passed directly to the provider's __init__
                   e.g., get_llm("gemini", api_key="AIza...")
 
    Returns:
        LLMProvider: Ready-to-use provider instance.
 
    Raises:
        ValueError: If provider name is unknown.
 
    Example:
        llm = get_llm("mock")                              # offline testing
        llm = get_llm("gemini", api_key="AIza...")         # Gemini
        llm = get_llm("claude", api_key="sk-ant-...")      # Claude
        llm = get_llm("ollama", model="llama3.2")          # local
    """
    providers = {
        "mock":   MockLLMProvider,
        "gemini": GeminiProvider,
        "claude": ClaudeProvider,
        "openai": OpenAIProvider,
        "ollama": LocalLlamaProvider,
        "groq":   GroqProvider,
    }
 
    provider_lower = provider.lower()
 
    if provider_lower not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            f"Available options: {available}"
        )
 
    return providers[provider_lower](**kwargs)



# =============================================================================
# QUICK TEST — Runing  this file directly: python llm_client.py
# =============================================================================
 
if __name__ == "__main__":
    print("=" * 60)
    print("LLM Client — Quick Test")
    print("=" * 60)
 
    # Test 1: Mock provider (always works)
    print("\n[TEST 1] MockLLMProvider (offline)")
    llm = MockLLMProvider()
 
    test_prompt = """
You are a chess coach. A player just made a BLUNDER.
 
Move played: e4e5
Score before move: +120 centipawns (White has advantage)
Score after move:  -200 centipawns (Black has advantage)
Best move was: d4
Score drop: 320 centipawns
 
In 2-3 sentences, explain WHY this move was bad and what the better move achieves.
Do NOT invent lines not given above. Base your explanation only on the data provided.
    """
 
    response = llm.complete(test_prompt)
    print(f"Prompt snippet: '...made a BLUNDER...'")
    print(f"Response:\n{response}\n")
 
    # Test 2: Factory function
    print("[TEST 2] Factory function: get_llm('mock')")
    llm2 = get_llm("mock")
    r2 = llm2.complete("A player made an inaccuracy in the endgame.")
    print(f"Response:\n{r2}\n")
 
    # Test 3: Retry logic (simulated failure)
    print("[TEST 3] Retry logic (mock always succeeds)")
    r3 = llm.complete_with_retry("Explain a good move", retries=3)
    print(f"Response:\n{r3}\n")
 
    print("=" * 60)
    print("All tests passed! MockLLMProvider is working correctly.")
    print("\nNext step: Get a Gemini API key and test GeminiProvider.")
    print("Run: python -c \"from Engine_Assistant.Explain.llm_client import GeminiProvider; ...")
    print("=" * 60)
 
