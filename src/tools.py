from __future__ import annotations


import json


from typing import Any, Callable


from urllib.error import HTTPError, URLError


from urllib.parse import urlencode


from urllib.request import Request, urlopen


_registry: dict[str, Callable[..., Any]] = {}


def _register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _registry[name] = fn
        return fn
    return decorator


def execute(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """등록된 도구 이름을 찾아 실행하고 결과를 돌려준다."""
    fn = _registry.get(name)
    if fn is None:
        print(f"[TOOL] Unknown tool: {name}")
        return {"ok": False, "message": f"Unknown tool: {name}"}
    try:
        print(f"[TOOL] Execute: {name}({arguments})")
        result = fn(**arguments)
        print(f"[TOOL] Result: {result}")
        return result
    except Exception as exc:
        print(f"[TOOL] Execution error ({name}): {exc}")
        return {"ok": False, "message": f"Execution failed: {exc}"}


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_user_profile",
        "description": "사용자의 프로필 정보를 조회한다. 이름, 나이, 생년월일, 직업, 거주지, 생활환경, 취미 등을 반환한다.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "list_diaries",
        "description": (
            "사용자의 최근 사건, 감정, 일상 기억을 장기기억에서 검색한다. "
            "특정 날짜가 있으면 diary_date='YYYY-MM-DD' 형식으로 지정할 수 있다. "
            "'어제 뭐했지', '최근에 어떤 일 있었지' 같은 질문에 사용한다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "diary_date": {
                    "type": "string",
                    "description": "검색 단서로 쓸 날짜다. YYYY-MM-DD 형식이다.",
                },
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "list_tasks",
        "description": (
            "사용자의 할 일을 조회한다. future_tasks는 날짜/시각을 보존한 미완료 일정이고, "
            "tasks는 기존 장기기억의 할 일이다. 같은 일정이 겹치면 future_tasks의 시각을 우선한다. "
            "특정 날짜의 할 일을 보고 싶으면 task_date='YYYY-MM-DD' 형식으로 지정한다. "
            "'오늘 할 일 뭐였지', '내일 일정 알려줘' 같은 질문에 사용한다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_date": {
                    "type": "string",
                    "description": "조회할 날짜다. YYYY-MM-DD 형식이다. 생략하면 미완료 일정을 가져온다.",
                },
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "recall_memory",
        "description": (
            "대화 중 더 자세한 과거 기억이 필요할 때 장기기억 서버에서 관련 fact를 검색한다. "
            "현재 prompt의 memory pack만으로 답하기 부족할 때 사용한다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색할 질문 또는 기억 단서.",
                },
                "k": {
                    "type": "integer",
                    "description": "가져올 기억 개수. 기본값은 5.",
                },
            },
            "required": ["query"],
        },
    },
]


_MEMORY_SERVER_URL = ""


_MEMORY_USER_ID = ""


_MEMORY_API_KEY = ""


_MEMORY_CHARACTER_ID = "study"


def set_memory_server_config(
    base_url: str,
    user_id: str,
    api_key: str,
    character_id: str = "study",
) -> None:
    """메모리 서버 접근 정보를 주입한다. 세션 시작 전에 호출해야 한다."""
    global _MEMORY_SERVER_URL, _MEMORY_USER_ID, _MEMORY_API_KEY, _MEMORY_CHARACTER_ID
    _MEMORY_SERVER_URL = base_url.rstrip("/")
    _MEMORY_USER_ID = user_id
    _MEMORY_API_KEY = api_key
    _MEMORY_CHARACTER_ID = character_id or "study"


@_register("get_user_profile")
def get_user_profile() -> dict[str, Any]:
    if not _MEMORY_SERVER_URL or not _MEMORY_USER_ID:
        return {"ok": False, "message": "메모리 서버가 설정되지 않았습니다."}
    url = f"{_MEMORY_SERVER_URL}/users/{_MEMORY_USER_ID}"
    try:
        req = Request(url, headers={"Accept": "application/json", "X-API-Key": _MEMORY_API_KEY, "User-Agent": "MindmateRaspberryPi/1.0"})
        with urlopen(req, timeout=10) as resp:
            data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        return {
            "ok": True,
            "user_name": data.get("user_name"),
            "age": data.get("age"),
            "birth_date": data.get("birth_date"),
            "job": data.get("job"),
            "location": data.get("location"),
            "living_info": data.get("living_info"),
            "habit": data.get("habit"),
        }
    except HTTPError as exc:
        return {"ok": False, "message": f"서버 오류: {exc.code}"}
    except URLError as exc:
        return {"ok": False, "message": f"서버 연결 실패: {exc}"}


def _memory_get(path: str, params: dict[str, Any]) -> Any:
    """메모리 서버 GET 요청 공통 처리."""
    query = f"?{urlencode(params)}" if params else ""
    url = f"{_MEMORY_SERVER_URL}{path}{query}"
    req = Request(
        url,
        headers={
            "Accept": "application/json",
            "X-API-Key": _MEMORY_API_KEY,
            "User-Agent": "MindmateRaspberryPi/1.0",
        },
    )
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _memory_post(path: str, payload: dict[str, Any]) -> Any:
    """메모리 서버 POST 요청 공통 처리."""
    url = f"{_MEMORY_SERVER_URL}{path}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-Key": _MEMORY_API_KEY,
            "User-Agent": "MindmateRaspberryPi/1.0",
        },
        method="POST",
    )
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


@_register("recall_memory")
def recall_memory(query: str, k: int = 5) -> dict[str, Any]:
    if not _MEMORY_SERVER_URL or not _MEMORY_USER_ID:
        return {"ok": False, "message": "memory server is not configured"}
    try:
        response: dict[str, Any] = _memory_post(
            f"/users/{_MEMORY_USER_ID}/recall-detail",
            {
                "query": query,
                "active_character_id": _MEMORY_CHARACTER_ID,
                "k": max(1, min(int(k or 5), 10)),
                "pool": 30,
            },
        )
        results: list[dict[str, Any]] = response.get("results") or []
        memories = [
            {
                "content": item.get("summary_for_prompt") or item.get("content"),
                "memory_type": item.get("memory_type"),
                "scope": item.get("scope"),
                "owner_character_id": item.get("owner_character_id"),
                "event_time": item.get("event_time"),
                "score": item.get("score"),
            }
            for item in results
        ]
        return {"ok": True, "count": len(memories), "memories": memories}
    except HTTPError as exc:
        return {"ok": False, "message": f"server error: {exc.code}"}
    except URLError as exc:
        return {"ok": False, "message": f"server connection failed: {exc}"}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


@_register("list_diaries")
def list_diaries(diary_date: str | None = None) -> dict[str, Any]:
    query = f"{diary_date} 사용자의 최근 사건과 감정 기억" if diary_date else "사용자의 최근 사건과 감정 기억"
    return recall_memory(query=query, k=5)


@_register("list_tasks")
def list_tasks(task_date: str | None = None) -> dict[str, Any]:
    if not _MEMORY_SERVER_URL or not _MEMORY_USER_ID:
        return {"ok": False, "message": "memory server is not configured"}
    try:
        params: dict[str, Any] = {"limit": 100, "character_id": _MEMORY_CHARACTER_ID}
        if task_date:
            params["task_date"] = task_date
        future_tasks = _memory_get(f"/users/{_MEMORY_USER_ID}/future-tasks", params)
        facts = _memory_get(f"/users/{_MEMORY_USER_ID}/facts", {"include_expired": "false"})
        tasks = [
            {"event_time": item.get("event_time"),
             "content": item.get("summary_for_prompt") or item.get("content")}
            for item in facts
            if item.get("memory_type") == "task"
            and item.get("selection_status") in {"candidate", "selected"}
            and (not task_date or item.get("event_time") == task_date)
        ]
        return {"ok": True, "count": len(tasks), "tasks": tasks[:10],
                "future_tasks": future_tasks, "future_task_count": len(future_tasks),
                "note": "세션 종료 후 추출된 일정입니다. needs_clarification=true이면 날짜/시각 확인이 필요합니다. 알람 재생은 아직 연결되지 않았습니다."}
    except HTTPError as exc:
        return {"ok": False, "message": f"server error: {exc.code}"}
    except URLError as exc:
        return {"ok": False, "message": f"server connection failed: {exc}"}
