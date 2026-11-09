"""Product entry point for the S4000 OpenAI-compatible server.

The actual implementation lives in :mod:`nanovllm.entrypoints.openai.api_server`
so both `s4000-server` and `python -m ...api_server` share one CLI and one set
of defaults.
"""
from nanovllm.entrypoints.openai.api_server import main

__all__ = ["main"]

if __name__ == "__main__":
    main()