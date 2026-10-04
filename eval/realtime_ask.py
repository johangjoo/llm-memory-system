"""T4 — Realtime API에 instructions + 한 질문을 보내고 응답 텍스트를 받는 평가용 헬퍼.

1차 구현: text-in / text-out (Realtime 모델이 메모리 블록을 받아 답하는 능력 측정).
매뉴얼이 가리키는 audio-in / audio-out → 전사 풀 루프는 후속(음성화 손실 별도 측정).

사용:
    out = ask_realtime(instructions, question)
    # out = {"answer", "prompt_tokens", "completion_tokens", "latency"}
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import websockets


REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-mini")


class RealtimeAskError(RuntimeError):
    """Realtime 호출 실패."""


async def _ask_realtime_async(
    instructions: str, question: str, model: str, timeout: float
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RealtimeAskError("OPENAI_API_KEY가 설정되지 않았습니다.")

    url = f"wss://api.openai.com/v1/realtime?model={model}"
    started = time.time()

    async with websockets.connect(
        url,
        additional_headers={"Authorization": f"Bearer {api_key}"},
        open_timeout=timeout,
    ) as ws:
        # 1) 세션 설정 (텍스트 전용, 오디오 끔)
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": model,
                "instructions": instructions,
                "output_modalities": ["text"],
            },
        }))
        # 2) 사용자 질문 입력
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": question}],
            },
        }))
        # 3) 응답 생성 요청
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {"output_modalities": ["text"]},
        }))

        # 4) 텍스트 델타·usage 수집
        text_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        done = False

        async def _drain():
            nonlocal prompt_tokens, completion_tokens, done
            async for raw in ws:
                event = json.loads(raw)
                etype = event.get("type", "")
                if etype == "response.output_text.delta":
                    text_parts.append(event.get("delta", ""))
                elif etype == "response.done":
                    usage = (event.get("response", {}) or {}).get("usage", {}) or {}
                    prompt_tokens = int(usage.get("input_tokens", 0) or 0)
                    completion_tokens = int(usage.get("output_tokens", 0) or 0)
                    done = True
                    return
                elif etype == "error":
                    raise RealtimeAskError(f"Realtime error: {event.get('error')}")

        try:
            await asyncio.wait_for(_drain(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise RealtimeAskError(f"Realtime 응답 타임아웃({timeout}s)") from exc

    return {
        "answer": "".join(text_parts),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency": time.time() - started,
    }


def ask_realtime(
    instructions: str,
    question: str,
    model: str | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Realtime API로 instructions + question을 한 번 보내고 텍스트 응답을 받는다(동기 래퍼)."""
    return asyncio.run(
        _ask_realtime_async(instructions, question, model or REALTIME_MODEL, timeout)
    )
