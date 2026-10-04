"""메모리 서버(FastAPI)와 통신하는 클라이언트.

MINDMATE_USER_ID에 설정된 user_id를 그대로 API에 사용한다.
(서버 DB의 user_id 필드와 동일한 값을 .env에 넣으면 된다.)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(slots=True, frozen=True)
class MemoryServerClient:
    base_url: str
    user_id: str
    api_key: str = ""
    character_id: str = "study"
    timeout_seconds: int = 10

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.user_id and self.api_key)

    async def verify(self) -> bool:
        """서버에 user_id가 실제로 존재하는지 확인한다."""
        if not self.enabled:
            return False
        try:
            await asyncio.to_thread(
                self._get_json,
                f"{self.base_url}/users/{self.user_id}",
            )
            return True
        except RuntimeError:
            return False

    async def get_memory_pack(
        self,
        character_id: str | None = None,
        *,
        shared_limit: int = 0,
        private_limit: int = 0,
        reflection_limit: int = 0,
        max_prompt_chars: int = 12000,
    ) -> dict[str, object] | None:
        """Fetch the compact memory pack for the active character."""
        if not self.enabled:
            return None
        active_character_id = character_id or self.character_id
        params = urlencode({
            "shared_limit": shared_limit,
            "private_limit": private_limit,
            "reflection_limit": reflection_limit,
            "max_prompt_chars": max_prompt_chars,
        })
        return await asyncio.to_thread(
            self._get_json,
            (
                f"{self.base_url}/users/{self.user_id}/characters/"
                f"{active_character_id}/memory-pack?{params}"
            ),
        )

    async def save_session(
        self,
        transcript: str | list[dict[str, str]],
        *,
        character_id: str | None = None,
        extract: bool = False,
    ) -> dict[str, object] | None:
        """Store a raw session and optionally trigger fact extraction."""
        if not self.enabled:
            return None
        turns = normalize_turns(transcript)
        if not turns:
            return None
        payload = {
            "session_date": datetime.now().date().isoformat(),
            "character_id": character_id or self.character_id,
            "turns": turns,
        }
        session = await asyncio.to_thread(
            self._post_json,
            f"{self.base_url}/users/{self.user_id}/sessions",
            payload,
        )
        if extract:
            session_id = str(session.get("session_id") or "")
            if session_id:
                session["extract_result"] = await asyncio.to_thread(
                    self._post_json,
                    f"{self.base_url}/users/{self.user_id}/sessions/{session_id}/extract",
                    {},
                )
        return session

    async def get_context(
        self,
        interaction_limit: int | None = None,
        diary_limit: int | None = None,
        task_limit: int | None = None,
        emotion_limit: int | None = None,
    ) -> dict[str, object] | None:
        """사용자 프로필과 최근 기록을 한 번에 가져온다.

        limit 인자를 비워두면 user_id 필터에 매칭되는 모든 문서를 받아온다.
        """
        _ = (interaction_limit, diary_limit, task_limit, emotion_limit)
        return await self.get_memory_pack()

    async def request_organize_now(
        self,
        transcript: str,
        interaction_date: str,
    ) -> dict[str, object] | None:
        """세션 종료 시 서버에 organize 분석을 즉시 실행하라고 요청한다.

        서버가 BackgroundTasks로 처리하므로 호출은 즉시 반환된다.
        Pi는 응답을 받자마자 셧다운해도 안전하다.
        """
        _ = interaction_date
        return await self.save_session(transcript, extract=True)

    async def ingest_text(self, raw_text: str) -> dict[str, object] | None:
        """사용자 발화 원문을 날짜별 interactions에 누적 저장한다."""
        return await self.save_session(raw_text, extract=False)

    def _get_json(self, url: str) -> dict[str, object]:
        request = Request(
            url=url,
            headers={
                "Accept": "application/json",
                "User-Agent": "MindmateRaspberryPi/1.0",
                "X-API-Key": self.api_key,
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Memory server request failed: {exc.code} {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Memory server request failed: {exc}") from exc

        result = json.loads(body)
        if not isinstance(result, dict):
            raise RuntimeError("Memory server returned a non-object JSON response.")
        return result

    def _post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            url=url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "MindmateRaspberryPi/1.0",
                "X-API-Key": self.api_key,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Memory server request failed: {exc.code} {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Memory server request failed: {exc}") from exc

        result = json.loads(body)
        if not isinstance(result, dict):
            raise RuntimeError("Memory server returned a non-object JSON response.")
        return result


def normalize_turns(transcript: str | list[dict[str, str]]) -> list[dict[str, str]]:
    if isinstance(transcript, str):
        text = transcript.strip()
        return [{"role": "user", "text": text}] if text else []

    turns: list[dict[str, str]] = []
    for turn in transcript:
        role = str(turn.get("role") or "").strip()
        text = str(turn.get("text") or "").strip()
        if role not in {"user", "assistant"} or not text:
            continue
        turns.append({"role": role, "text": text})
    return turns
