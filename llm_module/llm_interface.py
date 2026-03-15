"""
llm_module/llm_interface.py
----------------------------
Communicates with an LLM backend and returns raw generated text.

Supported backends
------------------
    mock        – Deterministic keyword engine (no external calls, great for tests/demos).
    ollama      – Local Ollama server (http://localhost:11434).
    openai      – OpenAI API (requires ``openai`` package + API key).
    huggingface – HuggingFace Inference API (requires ``requests`` + API key or free tier).

Public API
----------
    from llm_module.llm_interface import LLMInterface
    iface = LLMInterface()
    text  = iface.generate(prompt, backend="ollama", model="mistral")
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODELS = {
    "mock":        "mock-v1",
    "ollama":      "mistral",
    "openai":      "gpt-3.5-turbo",
    "huggingface": "mistralai/Mistral-7B-Instruct-v0.1",
}

_OLLAMA_URL    = "http://localhost:11434/api/generate"
_HF_API_URL    = "https://router.huggingface.co/{model}"


# ---------------------------------------------------------------------------
# LLMInterface
# ---------------------------------------------------------------------------

class LLMInterface:
    """
    Thin adapter layer between the TDDR pipeline and various LLM backends.

    All backends accept a *prompt* string and return a *raw text* string.
    Structured parsing is handled downstream by ``AnswerGenerator``.

    Usage
    -----
    >>> iface = LLMInterface()
    >>> text  = iface.generate(prompt, backend="mock")
    >>> text  = iface.generate(prompt, backend="ollama", model="mistral")
    """

    def generate(
        self,
        prompt: str,
        backend: str = "mock",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ) -> str:
        """
        Generate text from the given prompt.

        Parameters
        ----------
        prompt  : The fully constructed LLM prompt.
        backend : One of ``"mock"``, ``"ollama"``, ``"openai"``, ``"huggingface"``.
        model   : Model name/ID (uses sensible defaults per backend if None).
        api_key : API key (required for openai and huggingface backends).
        timeout : HTTP request timeout in seconds.

        Returns
        -------
        str
            Raw text output from the LLM (unparsed).

        Raises
        ------
        ValueError
            If an unknown backend is specified.
        RuntimeError
            If the LLM backend returns an error or is unreachable.
        """
        backend = backend.lower().strip()
        model   = model or _DEFAULT_MODELS.get(backend, "unknown")

        if backend == "mock":
            return self._mock(prompt)
        elif backend == "ollama":
            return self._ollama(prompt, model, timeout)
        elif backend == "openai":
            return self._openai(prompt, model, api_key, timeout)
        elif backend == "huggingface":
            return self._huggingface(prompt, model, api_key, timeout)
        else:
            raise ValueError(
                f"Unknown LLM backend: {backend!r}. "
                "Choose from: mock, ollama, openai, huggingface."
            )

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _mock(self, prompt: str) -> str:
        """
        Deterministic mock backend — no network calls, always succeeds.

        Extracts key signals from the prompt and synthesises a grounded
        answer using simple pattern matching. Good for demos and tests.
        """
        # Extract question
        question = ""
        for line in prompt.splitlines():
            if line.startswith("QUESTION:"):
                question = line.replace("QUESTION:", "").strip()
                break

        # Extract regulation names and sections from excerpts
        section_pattern = re.compile(
            r"Section\s+\d+\w*|Article\s+\d+\w*|Rule\s+\d+\w*|Clause\s+\d+\w*",
            re.IGNORECASE,
        )
        regulation_pattern = re.compile(r"\[(\d+)\]\s+(\w[\w_]*)\s+v")
        found_sections = list(dict.fromkeys(section_pattern.findall(prompt)))  # deduplicated
        found_regs     = [m.group(2) for m in regulation_pattern.finditer(prompt)]

        # Reference date
        ref_date = "not specified"
        for line in prompt.splitlines():
            if line.startswith("Reference date:"):
                ref_date = line.replace("Reference date:", "").strip()
                break

        # Determine if no chunks were retrieved
        no_chunks = "(none retrieved" in prompt

        if no_chunks:
            answer      = (
                "The retrieved regulation excerpts do not contain sufficient information "
                "to answer this question for the specified time period."
            )
            cited       = ""
            explanation = "No matching regulation chunks were retrieved from the vector store."
            confidence  = "low"
        else:
            regs_str = ", ".join(found_regs[:2]) if found_regs else "the applicable regulation"
            sec_str  = found_sections[0] if found_sections else "the relevant provision"
            answer = (
                f"Based on {regs_str}, {sec_str} addresses the query: \"{question}\". "
                f"The regulation in effect as of the reference date ({ref_date}) "
                f"sets out the applicable legal framework. "
                f"Please refer to the excerpts below for the precise statutory text."
            )
            cited       = ", ".join(found_sections[:4]) if found_sections else "General provision"
            explanation = (
                f"The retrieved excerpts from {regs_str} directly address the query. "
                f"The cited section(s) were active on the reference date ({ref_date})."
            )
            confidence  = "high" if len(found_sections) >= 1 else "medium"

        return (
            f"ANSWER: {answer}\n"
            f"CITED_SECTIONS: {cited}\n"
            f"EXPLANATION: {explanation}\n"
            f"CONFIDENCE: {confidence}"
        )

    def _ollama(self, prompt: str, model: str, timeout: int) -> str:
        """Call a local Ollama server."""
        payload = json.dumps({
            "model":  model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 512,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            _OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama not reachable at {_OLLAMA_URL}. "
                f"Is Ollama running? (ollama serve)  Error: {exc}"
            ) from exc

    def _openai(
        self,
        prompt: str,
        model: str,
        api_key: Optional[str],
        timeout: int,
    ) -> str:
        """Call the OpenAI Chat Completions API."""
        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI backend requires the `openai` package. "
                "Install it: pip install openai"
            ) from exc

        if not api_key:
            raise ValueError("OpenAI backend requires an api_key.")

        client = openai.OpenAI(api_key=api_key, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a legal compliance assistant."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    def _huggingface(
        self,
        prompt: str,
        model: str,
        api_key: Optional[str],
        timeout: int,
    ) -> str:
        """Call the HuggingFace Inference API."""
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFace backend requires the `requests` package. "
                "Install it: pip install requests"
            ) from exc

        url     = _HF_API_URL.format(model=model)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature":    0.2,
                "return_full_text": False,
            },
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(
                f"HuggingFace API error {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "").strip()
        raise RuntimeError(f"Unexpected HuggingFace API response format: {data!r}")
