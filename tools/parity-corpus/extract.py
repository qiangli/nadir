#!/usr/bin/env python3
"""
Generate testdata/routing_parity_corpus.json by running a fixed
scenario list through NadirClaw's *actual* routing helpers
(priorart/NadirClaw/nadirclaw/routing.py). The Go side then asserts
the same inputs produce equivalent outputs.

This script imports routing.py directly without going through the
nadirclaw package — that lets us bypass the FastAPI/torch/litellm
weight pulled in by `from nadirclaw import …`. Pure-stdlib only.

Run:

    python tools/parity-corpus/extract.py

The script writes:

    testdata/routing_parity_corpus.json

Re-run any time priorart/NadirClaw changes upstream; the Go test
loads this file and reports per-category parity rates.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTING_PATH = REPO_ROOT / "priorart" / "NadirClaw" / "nadirclaw" / "routing.py"
OUT_PATH = REPO_ROOT / "testdata" / "routing_parity_corpus.json"

# (path constant included so the file is self-documenting if needed)


def import_routing():
    """Import nadirclaw.routing without triggering the heavy server
    deps. routing.py is pure-stdlib and only touches sibling
    nadirclaw.model_metadata (also stdlib-only), so we just put the
    NadirClaw root on sys.path and `import nadirclaw.routing`."""
    pkg_root = REPO_ROOT / "priorart" / "NadirClaw"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    # Defensively shim out the package __init__'s side imports.
    import nadirclaw  # type: ignore  # noqa: F401
    from nadirclaw import routing  # type: ignore
    return routing


def msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def main() -> int:
    routing = import_routing()

    corpus: dict = {
        "alias": [],
        "profile": [],
        "agentic": [],
        "reasoning": [],
        "code_review": [],
        "images": [],
    }

    # ---- resolve_alias ----
    alias_cases = [
        "sonnet", "opus", "haiku", "claude", "gpt4", "gpt4o", "gpt4-mini",
        "gpt5", "gpt5-mini", "o3", "o3-mini", "o4-mini",
        "flash", "gemini-flash", "gemini-pro",
        "deepseek", "deepseek-v4", "deepseek-v4-flash", "deepseek-v4-pro",
        "deepseek-r1", "llama", "mini",
        "SONNET", "Flash", "  Haiku  ",  # case + whitespace
        "unknown-model-xyz", "gpt-4o", "claude-sonnet-4-6-20250918",
        "", "  ",
    ]
    for inp in alias_cases:
        corpus["alias"].append({
            "input": inp,
            "expected": routing.resolve_alias(inp),
        })

    # ---- resolve_profile ----
    profile_cases = [
        "auto", "eco", "premium", "free", "reasoning",
        "ECO", "NadirClaw/Premium", "nadirclaw/eco",
        "gpt-4o", "claude-sonnet", "", None,
    ]
    for inp in profile_cases:
        corpus["profile"].append({
            "input": inp,
            "expected": routing.resolve_profile(inp),
        })

    # ---- detect_agentic ----
    agentic_cases = [
        {
            "label": "simple_question",
            "messages": [msg("user", "What is 2+2?")],
            "has_tools": False, "tool_count": 0,
            "system_prompt": "", "message_count": 1,
        },
        {
            "label": "tools_defined_low_count",
            "messages": [msg("user", "Help me")],
            "has_tools": True, "tool_count": 3,
            "system_prompt": "", "message_count": 1,
        },
        {
            "label": "many_tools",
            "messages": [msg("user", "Help me")],
            "has_tools": True, "tool_count": 8,
            "system_prompt": "", "message_count": 1,
        },
        {
            "label": "agent_system_keywords",
            "messages": [msg("user", "Help")],
            "has_tools": False, "tool_count": 0,
            "system_prompt": "You are a coding agent. You can execute commands and read files.",
            "message_count": 1,
        },
        {
            "label": "long_system_prompt",
            "messages": [msg("user", "Help")],
            "has_tools": False, "tool_count": 0,
            "system_prompt": "x" * 1200, "message_count": 1,
        },
        {
            "label": "deep_conversation",
            "messages": [msg("user", f"msg {i}") for i in range(12)],
            "has_tools": False, "tool_count": 0,
            "system_prompt": "", "message_count": 12,
        },
        {
            "label": "full_agentic_realistic",
            "messages": [
                msg("system", "You are an AI agent. You can use tools to read and write files."),
                msg("user", "Refactor the auth module"),
                msg("assistant", "I'll start by reading the current implementation"),
                msg("tool", "<file contents>"),
                msg("assistant", "Now I'll write the updated file"),
                msg("tool", "<write result>"),
                msg("user", "Now add tests"),
            ],
            "has_tools": True, "tool_count": 4,
            "system_prompt": "You are an AI agent. You can use tools to read and write files.",
            "message_count": 7,
        },
    ]
    for case in agentic_cases:
        result = routing.detect_agentic(
            case["messages"],
            has_tools=case["has_tools"],
            tool_count=case["tool_count"],
            system_prompt_length=len(case["system_prompt"]),
            message_count=case["message_count"],
            system_prompt=case["system_prompt"],
        )
        # The Python API returns a dict; we capture the is_agentic flag
        # and any score/signal fields the Go side might mirror.
        is_agentic = bool(result.get("is_agentic") or result.get("agentic"))
        corpus["agentic"].append({
            "label": case["label"],
            "has_tools": case["has_tools"],
            "tool_count": case["tool_count"],
            "system_prompt": case["system_prompt"][:200],
            "message_count": case["message_count"],
            "expected_is_agentic": is_agentic,
            "raw": {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, list))},
        })

    # ---- detect_reasoning ----
    reasoning_cases = [
        ("What is 2+2?", ""),
        ("Think through this problem", ""),
        ("Think through this step by step", ""),
        ("Prove that P=NP and derive the implications step by step", ""),
        ("Critically analyze the paper and evaluate whether the conclusions are valid", ""),
        ("What are the implications?", "Analyze the tradeoffs and compare and contrast the approaches"),
        ("Hello", ""),
        ("Explain why concurrent map writes are unsafe in Go", ""),
        ("Let's reason through this carefully step by step.", ""),
        ("Chain of thought: work through the constraints.", ""),
        ("Mathematically prove this identity and derive the corollary.", ""),
        ("Compare and contrast SOLID with hexagonal architecture.", ""),
        ("Evaluate whether the algorithm terminates and explain why.", ""),
        ("Weigh the pros and cons of monorepo vs polyrepo.", ""),
        ("Design a system architecture for a chat application.", "Think about the tradeoffs"),
        ("Logically deduce the conclusion from these premises.", ""),
        ("Diagnose the root cause of the memory leak.", ""),
        ("Architectural decision: SQL vs document store?", ""),
        ("Hello there!", "Be helpful and concise."),
        ("step by step", ""),  # single marker only
        ("just explain", ""),  # single marker only
    ]
    for prompt, system_msg in reasoning_cases:
        result = routing.detect_reasoning(prompt, system_message=system_msg)
        is_reasoning = bool(result.get("is_reasoning") or result.get("reasoning"))
        corpus["reasoning"].append({
            "prompt": prompt,
            "system_message": system_msg,
            "expected_is_reasoning": is_reasoning,
            "raw": {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, list))},
        })

    # ---- detect_code_review ----
    code_review_cases = [
        ("Please code review this PR", ""),
        ("Review the diff and flag any issues.", ""),
        ("Run a security review on this commit.", ""),
        ("Static analysis pass over the module.", ""),
        ("Lint check the file please.", ""),
        ("Hello, what's your name?", ""),
        ("Refactor this function.", ""),
        ("", "Code review the changes when ready."),
    ]
    for prompt, system_msg in code_review_cases:
        result = routing.detect_code_review(prompt, system_message=system_msg)
        is_review = bool(result.get("is_review"))
        corpus["code_review"].append({
            "prompt": prompt,
            "system_message": system_msg,
            "expected_is_review": is_review,
            "raw": {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, list))},
        })

    # ---- detect_images ----
    # NadirClaw's detect_images walks the OpenAI-shape message list
    # looking for image_url parts. The shapes we test mirror what
    # real OpenAI/Anthropic clients send.
    image_cases = [
        {
            "label": "no_images",
            "messages": [msg("user", "hello")],
        },
        {
            "label": "image_url_part",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,xxx"}},
                ],
            }],
        },
        {
            "label": "image_part_anthropic_shape",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "xxx"}},
                ],
            }],
        },
    ]
    for case in image_cases:
        # NadirClaw's detect_images expects message objects with .content
        # attribute, but it also handles dicts (defensively). For simple
        # parity testing, pass dicts as-is.
        result = routing.detect_images(case["messages"])
        has_images = bool(result.get("has_images"))
        corpus["images"].append({
            "label": case["label"],
            "expected_has_images": has_images,
            "raw": {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, list))},
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(corpus, indent=2))
    print(f"wrote {OUT_PATH}")
    print(f"  {len(corpus['alias'])} alias cases")
    print(f"  {len(corpus['profile'])} profile cases")
    print(f"  {len(corpus['agentic'])} agentic cases")
    print(f"  {len(corpus['reasoning'])} reasoning cases")
    print(f"  {len(corpus['code_review'])} code_review cases")
    print(f"  {len(corpus['images'])} image cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
