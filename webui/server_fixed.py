"""
Fixed server that delays mai_dx import to avoid Windows hang
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from queue import Queue
import json
from fastapi import Body

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from loguru import logger
import warnings

# Silence warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure Windows compatibility before any other imports
if sys.platform == "win32":
    sys.path.insert(0, str(Path(__file__).parent))
    from swarms_compat import ensure_windows_compatibility
    ensure_windows_compatibility()

# DO NOT import mai_dx here - it will hang!
# We'll import it lazily when actually needed

logger.info("✅ Server module loaded (mai_dx import delayed)")

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

class StartCaseRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)
    initial_presentation: str
    full_case_details: Optional[str] = ""
    ground_truth: Optional[str] = ""
    requested_model: Optional[str] = Field(default=None, alias="model_name")
    max_iterations: int = 5
    mode: str = "no_budget"
    interactive: bool = False

@dataclass
class CaseRunner:
    case_id: str
    params: StartCaseRequest
    loop: asyncio.AbstractEventLoop
    input_queue: Queue = Queue()

    def __post_init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self._thread: Optional[threading.Thread] = None
        self._sink_id: Optional[int] = None

    def _log_sink(self, message) -> None:
        try:
            record = message.record
            payload = {
                "type": "log",
                "level": record["level"].name,
                "time": str(record["time"]),
                "message": record["message"],
            }
            asyncio.run_coroutine_threadsafe(self.queue.put(payload), self.loop)
        except Exception as e:
            print(f"[webui] log sink error: {e}")

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # Lazy import mai_dx here, in the thread, to avoid blocking main thread
        try:
            logger.info("Importing MAI-DxO backend (may take a moment)...")
            from mai_dx.main import MaiDxOrchestrator
            logger.info("✅ MAI-DxO backend imported successfully")
        except Exception as e:
            logger.error(f"Failed to import MAI-DxO: {e}")
            asyncio.run_coroutine_threadsafe(
                self.queue.put({"type": "error", "message": f"Failed to load diagnostic engine: {e}"}), 
                self.loop
            )
            return

        self._sink_id = logger.add(self._log_sink, level="INFO")
        try:
            orchestrator = MaiDxOrchestrator(
                model_name=(self.params.requested_model or "gpt-4o"),
                max_iterations=self.params.max_iterations,
                mode=self.params.mode,
                interactive=self.params.interactive,
            )

            gen = orchestrator.run_gen(
                initial_case_info=self.params.initial_presentation,
                full_case_details=self.params.full_case_details or "",
                ground_truth_diagnosis=self.params.ground_truth or "",
            )

            response = None
            while True:
                try:
                    update = gen.send(response)
                    response = None
                    asyncio.run_coroutine_threadsafe(self.queue.put(update), self.loop)

                    if update.get("type") == "pause":
                        user_input = self.input_queue.get()
                        response = user_input

                except StopIteration as e:
                    result = e.value
                    payload = {
                        "type": "result",
                        "final_diagnosis": getattr(result, "final_diagnosis", None),
                        "ground_truth": getattr(result, "ground_truth", None),
                        "accuracy_score": getattr(result, "accuracy_score", None),
                        "accuracy_reasoning": getattr(result, "accuracy_reasoning", None),
                        "total_cost": getattr(result, "total_cost", None),
                        "iterations": getattr(result, "iterations", None),
                    }
                    asyncio.run_coroutine_threadsafe(self.queue.put(payload), self.loop)
                    break
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                self.queue.put({"type": "error", "message": str(e)}), self.loop
            )
        finally:
            if self._sink_id is not None:
                try:
                    logger.remove(self._sink_id)
                except:
                    pass
            asyncio.run_coroutine_threadsafe(self.queue.put({"type": "done"}), self.loop)

app = FastAPI(title="MAI-DxO Prototype Web UI")

# Serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory store of running cases
CASES: Dict[str, CaseRunner] = {}

# --- API Key Management ---
KEYS_FILE = BASE_DIR / "api_keys.json"

def load_keys_from_file():
    if KEYS_FILE.exists():
        try:
            with open(KEYS_FILE, 'r') as f:
                keys = json.load(f)
                for key_name, key_value in keys.items():
                    if key_value:
                        os.environ[key_name] = key_value
            logger.info("Loaded API keys from api_keys.json")
        except Exception as e:
            logger.warning(f"Could not load keys from api_keys.json: {e}")

@app.on_event("startup")
async def startup_event():
    load_keys_from_file()
    logger.info("🚀 MAI-DxO Web UI started successfully")
    logger.info("   Note: MAI-DxO backend will load on first use to avoid Windows issues")

@app.get("/")
async def get_index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)

@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)

@app.post("/api/start_case")
async def start_case(payload: StartCaseRequest) -> JSONResponse:
    case_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    
    # Determine effective model
    effective_model = payload.requested_model or "gpt-4o"
    effective = StartCaseRequest(
        initial_presentation=payload.initial_presentation,
        full_case_details=payload.full_case_details,
        ground_truth=payload.ground_truth,
        requested_model=effective_model,
        max_iterations=payload.max_iterations,
        mode=payload.mode,
        interactive=payload.interactive,
    )
    
    runner = CaseRunner(case_id=case_id, params=effective, loop=loop)
    CASES[case_id] = runner
    runner.start()
    
    return JSONResponse({
        "case_id": case_id, 
        "status": "started",
        "effective_model": effective_model, 
        "backend": "real"
    })

@app.post("/api/save_keys")
async def save_keys(payload: Dict[str, str] = Body(...)):
    try:
        valid_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
        keys_to_save = {k: v for k, v in payload.items() if k in valid_keys}
        with open(KEYS_FILE, 'w') as f:
            json.dump(keys_to_save, f, indent=4)
        load_keys_from_file()
        return JSONResponse({"status": "success", "message": "API keys saved."})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.websocket("/ws/{case_id}")
async def ws_case_updates(websocket: WebSocket, case_id: str):
    await websocket.accept()
    runner = CASES.get(case_id)
    if not runner:
        await websocket.send_json({"type": "error", "message": "Invalid case_id"})
        await websocket.close()
        return
    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                if msg.get("type") == "input" and runner:
                    runner.input_queue.put(msg.get("data"))
            except asyncio.TimeoutError:
                pass
            if not runner.queue.empty():
                update = await runner.queue.get()
                await websocket.send_json(update)
                if update.get("type") == "done":
                    break
    except WebSocketDisconnect:
        return

# Stub endpoints for agent prompts (not critical)
@app.get("/api/agent_prompts")
async def get_agent_prompts() -> JSONResponse:
    return JSONResponse({"prompts": {}})

@app.post("/api/update_agent_prompt")
async def update_agent_prompt(payload: Dict[str, str]) -> JSONResponse:
    return JSONResponse({"status": "updated"})
