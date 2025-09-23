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

# Silence benign runtime and pydantic warnings for cleaner console output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure Windows compatibility before importing MAI-DxO
if sys.platform == "win32":
    # Import the compatibility layer
    sys.path.insert(0, str(Path(__file__).parent))
    from swarms_compat import ensure_windows_compatibility
    ensure_windows_compatibility()

# Import orchestrator from the existing package
try:
    from mai_dx.main import MaiDxOrchestrator
    logger.info("✅ Real MAI-DxO backend loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load MAI-DxO backend: {e}")
    raise


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

	# Log sink that forwards Loguru records to the async queue
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
		except Exception as e:  # pragma: no cover - defensive
			print(f"[webui] log sink error: {e}")

	def start(self) -> None:
		self._thread = threading.Thread(target=self._run, daemon=True)
		self._thread.start()

	def _run(self) -> None:
		# Attach log sink
		self._sink_id = logger.add(self._log_sink, level="INFO")
		try:
			orchestrator = MaiDxOrchestrator(
				model_name=(self.params.requested_model or "gpt-4o"),
				max_iterations=self.params.max_iterations,
				mode=self.params.mode,
				interactive=self.params.interactive,
			)

			# --- Modified generator handling for interactivity ---
			gen = orchestrator.run_gen(
				initial_case_info=self.params.initial_presentation,
				full_case_details=self.params.full_case_details or "",
				ground_truth_diagnosis=self.params.ground_truth or "",
			)

			response = None
			while True:
				try:
					# The `send()` method resumes the generator and sends a value in.
					# `next()` is equivalent to `send(None)`.
					update = gen.send(response)
					response = None # Reset response after sending

					asyncio.run_coroutine_threadsafe(self.queue.put(update), self.loop)

					if update.get("type") == "pause":
						# Wait for the UI to provide input via the input_queue
						user_input = self.input_queue.get() # This is a blocking call
						response = user_input # Set the response to be sent on the next loop

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
		except Exception as e:  # pragma: no cover
			asyncio.run_coroutine_threadsafe(
				self.queue.put({"type": "error", "message": str(e)}), self.loop
			)
		finally:
			if self._sink_id is not None:
				try:
					logger.remove(self._sink_id)
				except Exception:
					pass
			asyncio.run_coroutine_threadsafe(self.queue.put({"type": "done"}), self.loop)


app = FastAPI(title="MAI-DxO Prototype Web UI")

# Serve the static UI
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/agent_prompts")
async def get_agent_prompts() -> JSONResponse:
    prompts = MaiDxOrchestrator.get_agent_prompts()
    return JSONResponse({"prompts": prompts})

@app.post("/api/update_agent_prompt")
async def update_agent_prompt(payload: Dict[str, str]) -> JSONResponse:
    agent_name = payload.get("agent_name")
    new_prompt = payload.get("new_prompt")
    if not agent_name or not new_prompt:
        return JSONResponse({"error": "Missing agent_name or new_prompt"}, status_code=400)
    MaiDxOrchestrator.update_agent_prompt(agent_name, new_prompt)
    return JSONResponse({"status": "updated"})


def _resolve_model_choice(requested: Optional[str]) -> tuple[str, Optional[str]]:
	requested = requested or "gpt-4o"
	has_openai = bool(os.getenv("OPENAI_API_KEY"))
	has_gemini = bool(os.getenv("GEMINI_API_KEY"))
	has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

	def provider_for(model: str) -> str:
		m = model.lower()
		if "gemini" in m:
			return "gemini"
		if "claude" in m:
			return "anthropic"
		return "openai"

	provider = provider_for(requested)
	note: Optional[str] = None

	if provider == "openai" and not has_openai:
		if has_gemini:
			note = "OPENAI_API_KEY not found. Falling back to Gemini."
			return "gemini/gemini-2.5-flash", note
		if has_anthropic:
			note = "OPENAI_API_KEY not found. Falling back to Claude."
			return "claude-3-5-sonnet-20241022", note
		note = "No provider API keys found. Please set OPENAI_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY."
		return requested, note

	if provider == "gemini" and not has_gemini:
		if has_openai:
			note = "GEMINI_API_KEY not found. Falling back to OpenAI."
			return "gpt-4o", note
		if has_anthropic:
			note = "GEMINI_API_KEY not found. Falling back to Claude."
			return "claude-3-5-sonnet-20241022", note
		note = "No provider API keys found. Please set GEMINI_API_KEY or choose OpenAI with an OPENAI_API_KEY."
		return requested, note

	if provider == "anthropic" and not has_anthropic:
		if has_openai:
			note = "ANTHROPIC_API_KEY not found. Falling back to OpenAI."
			return "gpt-4o", note
		if has_gemini:
			note = "ANTHROPIC_API_KEY not found. Falling back to Gemini."
			return "gemini/gemini-2.5-flash", note
		note = "No provider API keys found. Please set ANTHROPIC_API_KEY or choose a model with a matching key."
		return requested, note

	return requested, None


@app.post("/api/start_case")
async def start_case(payload: StartCaseRequest) -> JSONResponse:
	case_id = str(uuid.uuid4())
	loop = asyncio.get_running_loop()
	effective_model, note = _resolve_model_choice(payload.requested_model)
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
	return JSONResponse({"case_id": case_id, "status": "started", "effective_model": effective_model, "note": note, "backend": "real"})


# In-memory store of running cases
CASES: Dict[str, CaseRunner] = {}

# --- API Key Management ---

# Define the path to a new file that will store the keys
KEYS_FILE = BASE_DIR / "api_keys.json"

def load_keys_from_file():
    """Loads keys from the JSON file and sets them as environment variables for this session."""
    if KEYS_FILE.exists():
        try:
            with open(KEYS_FILE, 'r') as f:
                keys = json.load(f)
                for key_name, key_value in keys.items():
                    if key_value:  # Only set if the key is not empty
                        os.environ[key_name] = key_value
            logger.info("Loaded API keys from api_keys.json")
        except Exception as e:
            logger.warning(f"Could not load keys from api_keys.json: {e}")

@app.on_event("startup")
async def startup_event():
    """On server startup, load any saved API keys."""
    load_keys_from_file()

@app.post("/api/save_keys")
async def save_keys(payload: Dict[str, str] = Body(...)):
    """Receives keys from the UI and saves them to the JSON file."""
    try:
        valid_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
        keys_to_save = {k: v for k, v in payload.items() if k in valid_keys}
        with open(KEYS_FILE, 'w') as f:
            json.dump(keys_to_save, f, indent=4)
        
        # Immediately load the new keys into the environment
        load_keys_from_file()
        
        return JSONResponse({"status": "success", "message": "API keys saved."})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# --- End of API Key Management ---


@app.get("/")
async def get_index() -> HTMLResponse:
	html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
	return HTMLResponse(content=html)

# Silence favicon 404s (optional; no icon yet)
@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


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