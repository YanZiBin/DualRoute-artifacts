# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# Modified by the DualRoute authors for the prompt-excluded release; numerical/parser/guard scope is documented in docs/REPRODUCTION_SCOPE.md.
"""Fast Path v1 routing and execution helpers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib import error as urllib_error
from urllib import request as urllib_request

from phone_agent.config.fast_path_apps import (
    FAST_PATH_APP_ALIASES,
)
from phone_agent.config.route_prompt import get_route_prompt
from phone_agent.device_factory import get_device_factory
from phone_agent.semantic_contract import (
    ROUTE_FAST_PATH,
    ROUTE_REJECT,
    ROUTE_SLOW_PATH,
)


class FastPathDecisionKind(str, Enum):
    """Possible Fast Path v1 routing outcomes."""

    FAST_LAUNCH = "fast_launch"
    FALLBACK_SLOW = "fallback_slow"
    UNSUPPORTED_OR_CLARIFY = "unsupported_or_clarify"


class FastPathParserStatus(str, Enum):
    """Typed parser outcomes consumed by the final route combiner."""

    VALID_ATOMIC_LAUNCH = "valid_atomic_launch"
    KNOWN_COMPLEX_TASK = "known_complex_task"
    UNKNOWN_TARGET = "unknown_target"
    UNSAFE_TARGET = "unsafe_target"
    AMBIGUOUS_TARGET = "ambiguous_target"
    MISSING_TARGET = "missing_target"
    UNSUPPORTED_CONTRACT = "unsupported_contract"


@dataclass(frozen=True)
class FastPathParseResult:
    """Structured result of parsing the deterministic Launch contract."""

    status: FastPathParserStatus
    app_name: str | None = None
    matched_app_count: int = 0
    launch_intent_count: int = 0


@dataclass
class FastPathDecision:
    """Normalized Fast Path routing result."""

    kind: FastPathDecisionKind
    app_name: str | None = None
    route_label: str | None = None
    reason: str | None = None
    message: str | None = None
    raw_prediction: str | None = None
    classifier_label: str | None = None
    classifier_outcome: str | None = None
    parser_status: str | None = None
    guard_outcome: str | None = None
    matched_app_count: int = 0
    launch_intent_count: int = 0

    @property
    def is_fast_path(self) -> bool:
        """Whether this decision should execute locally."""
        return self.kind == FastPathDecisionKind.FAST_LAUNCH

    @property
    def final_route(self) -> str:
        """Return the stable experiment-log route name for this decision."""
        if self.kind == FastPathDecisionKind.FAST_LAUNCH:
            return ROUTE_FAST_PATH
        if self.kind == FastPathDecisionKind.FALLBACK_SLOW:
            return ROUTE_SLOW_PATH
        return ROUTE_REJECT


class LocalRouteModel:
    """Lazy wrapper around the locally fine-tuned Model A route classifier."""

    @property
    def SYSTEM_PROMPT(self) -> str:
        return get_route_prompt()

    def __init__(self, base_model_path: str | Path, model_a_path: str | Path):
        self.base_model_path = Path(base_model_path)
        self.model_a_path = Path(model_a_path)
        self._model = None
        self._tokenizer = None

    def predict_label(self, text: str) -> str:
        """Predict the 0/1/2 route label for a user instruction."""
        return self.predict(text)["label"]

    def predict(self, text: str) -> dict[str, str]:
        """Predict route label and preserve raw generated output for debugging."""
        system_prompt = self.SYSTEM_PROMPT
        self._ensure_loaded()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model_device)

        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generated_ids = outputs[0, inputs.input_ids.shape[1] :]
        raw_prediction = self._tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        match = re.search(r"\b([012])\b", raw_prediction)
        if match:
            return {"label": match.group(1), "raw_prediction": raw_prediction}

        if raw_prediction in {"0", "1", "2"}:
            return {"label": raw_prediction, "raw_prediction": raw_prediction}

        raise RoutePredictionError(raw_prediction)

    def load(self) -> None:
        """Eagerly load the local route model into memory."""
        get_route_prompt()
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        if not self.base_model_path.exists():
            raise FileNotFoundError(
                f"Fast Path base model path not found: {self.base_model_path}"
            )
        if not self.model_a_path.exists():
            raise FileNotFoundError(
                f"Fast Path Model A path not found: {self.model_a_path}"
            )

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Fast Path local router dependencies are missing. "
                "Please install torch, transformers, and peft."
            ) from exc

        self._torch = torch

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.base_model_path),
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if torch.cuda.is_available():
            base_model = AutoModelForCausalLM.from_pretrained(
                str(self.base_model_path),
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                str(self.base_model_path),
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )

        self._model = PeftModel.from_pretrained(
            base_model,
            str(self.model_a_path),
        )
        self._model.eval()

        if not torch.cuda.is_available():
            self._model.to("cpu")

        self._model_device = next(self._model.parameters()).device


class RemoteRoutePredictor:
    """Call a long-lived local Fast Path route service over HTTP."""

    def __init__(self, service_url: str, timeout_seconds: float = 30.0):
        normalized = service_url.rstrip("/")
        if not normalized:
            raise ValueError("Fast Path route service URL cannot be empty.")
        self.service_url = normalized
        self.timeout_seconds = timeout_seconds

    def predict_label(self, text: str) -> str:
        """Request a route label from the local Fast Path route service."""
        payload = json.dumps({"text": text}).encode("utf-8")
        request = urllib_request.Request(
            f"{self.service_url}/route",
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Fast Path route service returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(
                f"Fast Path route service unavailable at {self.service_url}: {exc.reason}"
            ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Fast Path route service returned invalid JSON: {body!r}"
            ) from exc

        label = parsed.get("label")
        if label in {"0", "1", "2"}:
            return label

        raise RuntimeError(f"Fast Path route service returned invalid label: {parsed!r}")


class _FastPathRouteServiceHandler(BaseHTTPRequestHandler):
    """Handle health and route requests for the local Fast Path route service."""

    server_version = "FastPathRouteService/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/route":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return

        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_text"})
            return

        predictor = self.server.predictor  # type: ignore[attr-defined]
        memory_before = _read_process_memory_snapshot()
        try:
            prediction = _normalize_route_prediction(predictor(text))
        except RoutePredictionError as exc:
            memory_after = _read_process_memory_snapshot()
            _log_route_debug_event(
                text=text,
                status="prediction_failed",
                label=None,
                raw_prediction=exc.raw_prediction,
                memory_before=memory_before,
                memory_after=memory_after,
            )
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": "prediction_failed",
                    "detail": str(exc),
                    "raw_prediction": exc.raw_prediction,
                    "pid": os.getpid(),
                    "memory_before": memory_before,
                    "memory_after": memory_after,
                },
            )
            return
        except Exception as exc:  # pragma: no cover - defensive branch
            memory_after = _read_process_memory_snapshot()
            _log_route_debug_event(
                text=text,
                status="prediction_failed",
                label=None,
                raw_prediction=None,
                memory_before=memory_before,
                memory_after=memory_after,
                error=str(exc),
            )
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": "prediction_failed",
                    "detail": str(exc),
                    "pid": os.getpid(),
                    "memory_before": memory_before,
                    "memory_after": memory_after,
                },
            )
            return

        memory_after = _read_process_memory_snapshot()
        _log_route_debug_event(
            text=text,
            status="ok",
            label=prediction["label"],
            raw_prediction=prediction["raw_prediction"],
            memory_before=memory_before,
            memory_after=memory_after,
        )
        self._write_json(
            HTTPStatus.OK,
            {
                "label": prediction["label"],
                "raw_prediction": prediction["raw_prediction"],
                "pid": os.getpid(),
                "memory_before": memory_before,
                "memory_after": memory_after,
                "memory": memory_after,
            },
        )

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        """Silence the default HTTP request logs."""
        return

    def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class FastPathRouteService:
    """Small HTTP server that keeps the Fast Path route model loaded once."""

    def __init__(self, predictor: Callable[[str], str], host: str = "127.0.0.1", port: int = 8765):
        self._server = ThreadingHTTPServer((host, port), _FastPathRouteServiceHandler)
        self._server.predictor = predictor  # type: ignore[attr-defined]

    @property
    def host(self) -> str:
        return str(self._server.server_address[0])

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        host = self.host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def serve_fast_path_route_service(
    base_model_path: str | Path,
    model_a_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Start a long-lived local Fast Path route service process."""
    local_route_model = LocalRouteModel(base_model_path=base_model_path, model_a_path=model_a_path)
    local_route_model.load()
    service = FastPathRouteService(local_route_model.predict_label, host=host, port=port)

    print("Fast Path route service is ready.")
    print(f"  Base URL: {service.base_url}")
    print(f"  Health:   {service.base_url}/health")
    print(f"  Route:    {service.base_url}/route")
    print("Press Ctrl+C to stop.")

    try:
        service.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Fast Path route service...")
    finally:
        service.shutdown()


class RoutePredictionError(ValueError):
    """Prediction returned unexpected free-form text instead of 0/1/2."""

    def __init__(self, raw_prediction: str):
        self.raw_prediction = raw_prediction
        super().__init__(f"Unexpected local route model output: {raw_prediction!r}")


def _normalize_route_prediction(prediction: object) -> dict[str, str]:
    """Normalize different predictor return shapes into a stable service payload."""
    if isinstance(prediction, dict):
        label = prediction.get("label")
        raw_prediction = prediction.get("raw_prediction")
        if label in {"0", "1", "2"} and isinstance(raw_prediction, str):
            return {"label": str(label), "raw_prediction": raw_prediction}

    if isinstance(prediction, str) and prediction in {"0", "1", "2"}:
        return {"label": prediction, "raw_prediction": prediction}

    raise RuntimeError(f"Fast Path route service returned invalid label: {prediction!r}")


def _read_process_memory_snapshot() -> dict[str, int]:
    """Read current process memory stats from /proc for lightweight route debugging."""
    status_path = Path("/proc/self/status")
    metrics = {"vmsize_kb": -1, "vmrss_kb": -1, "threads": -1}

    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmSize:"):
                metrics["vmsize_kb"] = int(line.split()[1])
            elif line.startswith("VmRSS:"):
                metrics["vmrss_kb"] = int(line.split()[1])
            elif line.startswith("Threads:"):
                metrics["threads"] = int(line.split()[1])
    except Exception:
        return metrics

    return metrics


def _log_route_debug_event(
    *,
    text: str,
    status: str,
    label: str | None,
    raw_prediction: str | None,
    memory_before: dict[str, int],
    memory_after: dict[str, int],
    error: str | None = None,
) -> None:
    """Emit a compact JSON debug line for each route request."""
    event = {
        "event": "fast_path_route_debug",
        "pid": os.getpid(),
        "status": status,
        "text": text,
        "label": label,
        "raw_prediction": raw_prediction,
        "memory_before": memory_before,
        "memory_after": memory_after,
    }
    if error is not None:
        event["error"] = error

    print(json.dumps(event, ensure_ascii=False), flush=True)


class FastPathRouter:
    """Keep the learned route model primary and apply a one-way safety guard."""

    _HARD_GUARD_REJECT_STATUSES = frozenset(
        {
            FastPathParserStatus.UNKNOWN_TARGET,
            FastPathParserStatus.UNSAFE_TARGET,
            FastPathParserStatus.AMBIGUOUS_TARGET,
            FastPathParserStatus.MISSING_TARGET,
        }
    )

    # Complete state transitions; share their predicates with prohibition and
    # deferral checks so recognizing a new launch never bypasses those checks.
    _STATE_LAUNCH_VERB = (
        r"切过去|切换到最前面|停在最上层|回到屏幕上|"
        r"(?:重新)?出现在(?:眼前|手机前面)|成为当前应用|叫出来|"
        r"叫到屏幕最前面|调到眼前|重新显示在前台|调回来|(?:界面)?调出来"
    )
    _NEW_FORWARD_LAUNCH_VERB = r"切回|叫出|显示一下"
    _REFERENTIAL_CHOICE_CONTRACT = (
        r"\s*(?:"
        r"能用{target}也能用{target}[，,；;]\s*(?:此次|这次)只(?:启动|打开)你选中的那一个|"
        r"(?:开|打开){target}还是(?:开|打开){target}都可以[，,]\s*你决定其一[，,]\s*"
        r"另一个无需(?:启动|打开)"
        r")[。！？!?]?\s*"
    )
    _LOCAL_CONTENT_TAIL = re.compile(
        r"(?:后续)?(?:刷视频(?:的动作)?|车次查询)(?:先|暂时)?不做|"
        r"路线(?:先|暂时)?不规划|问题(?:先|暂时)?不输入|"
        r"播放按钮(?:先)?不要按|不进行点单|不阅读邮件|不要点文章|"
        r"别选剧集|不再点击游戏里的按钮|别去点笔记正文|"
        r"商品详情(?:晚些|稍后)再看|先不要刷新或查看热搜"
    )
    _NEGATED_CONTENT_CONTINUATION = re.compile(
        r"(?:不用|不要|别|无需|不需要|不必|不)"
        r"(?:继续)?(?:查|翻|规划)$|"
        r"(?:不用|不要|别|无需|不需要|不必)"
        r"(?:选择|播放|刷新|查看|滑动)(?:节目|视频|歌曲|内容)?或(?:查)?$|"
        r"(?:不用|不要|别|无需|不需要|不必)继续查$|"
        r"(?:不用|不要|别|无需|不需要|不必)(?:帮我|替我|为我|碰)$"
    )
    _NATURAL_LAUNCH_REFERENCE = re.compile(
        r"(?:APP|它|这个应用|应用本身|应用首页|应用|前台|首页|初始页面|"
        r"订单|联系人|会话|消息|文档|内容|支付|其他功能|任何回答|任何频道|任何商户详情|报道|小说)"
        r"(?=$|[\s，,。！？!?；;]|后|就|便|或|即可|这一步)|"
        r"后|以后|即可|便完成|不叫车|的(?:位置|页面|主页面|首页|状态)|到初始页面|这一步"
    )
    _ACTION_CLAUSE_SEPARATOR = re.compile(
        r"[，,。！？!?；;：:\n]|但是|但|不过|可是|然而|然后|接着|随后|并且|并(?!非)|同时"
    )

    _LAUNCH_CUE_PATTERN = re.compile(
        r"打开|启动|进入|开一下|开起来|开出来|快开|起起起|切到|切换到|"
        r"切一下|切过去|切回|转到|唤起|调出|调出来|跳到|跳转到|到前台|"
        r"最前面|最上层|点开|整开|弄开|搞开|叫出|显示一下"
    )
    _FORWARD_TARGET_LAUNCH_CUE_PATTERN = re.compile(
        r"打开|启动|进入|开一下|开下|开个|快开|切到|切换到|切换至|切回|"
        r"转到|唤起|调出来|跳到|跳转到|点开|整开|弄开|搞开|叫出|显示一下"
    )
    _DIRECT_LAUNCH_CUE_PATTERN = re.compile(
        r"打开|启动|进入|开一下|开下|开个|开起来|开出来|快开|切到|切换到|"
        r"切换至|切回|切一下|切过去|转到|唤起|调出来|调到前台|调至最前面|"
        r"放到最前面|放到最上层|移到前台|移到最前面|移至最前面|回到前台|"
        r"到前台|跳到|跳转到|点开|整开|弄开|搞开"
    )
    _DEFERRED_LAUNCH_PATTERN = re.compile(
        r"(?:(?:等(?:一下|一会|会儿)|稍后|晚些|晚点|待会|过(?:一会|会儿)|明天|下次|之后)|"
        r"等我[^，,。！？!?；;]{0,8}(?:后|再)|"
        r"(?:周末|下班|放学|到家|到公司)[^，,。！？!?；;]{0,6}(?:后|了)?(?:再|就))"
        rf"[^，,。！？!?；;]{{0,12}}(?:打开|启动|进入|切到|唤起|{_NEW_FORWARD_LAUNCH_VERB}|{_STATE_LAUNCH_VERB})"
        r"|[，,；;]\s*(?:过半小时|过一小时|稍后|晚些|晚点|待会|明天|后天|等我[^，,。！？!?；;]{1,12})"
        r"(?:再|才)(?:执行|操作|启动|打开)"
    )
    _DEFERRED_OR_META_LAUNCH_PATTERN = re.compile(
        _DEFERRED_LAUNCH_PATTERN.pattern
        + r"|你觉得|我想知道|是否|合适吗|该不该|要不要"
    )
    _LAUNCH_CONTRACT_PATTERNS = tuple(
        re.compile(pattern)
        for pattern in (
            rf"(?:{_NEW_FORWARD_LAUNCH_VERB})\s*APP(?:本身)?",
            rf"(?:把|将|让)?\s*APP\s*(?:{_STATE_LAUNCH_VERB})",
            r"(?:打开|启动|进入|进一下|开一下|开下|开个|点开|开|快开|切到|切换到|切换至|转到|唤起|调出|调出来|跳到|跳转到)\s*(?:一下)?\s*APP",
            r"(?:先)?去\s*APP(?=$|[\s，,。！？!?；;]|里|中|内|查看|搜索|查找)",
            r"(?:把|将)\s*APP\s*(?:给我)?\s*(?:打开|启动|开起来|开出来|开一下|开下|整开|点开|弄开(?:一下)?|搞开)(?:了)?",
            r"APP\s*(?:现在|马上)?\s*(?:给我)?\s*(?:打开|启动|进入|进一下|开起来|开出来|开一下|开下|快开|切一下|转到|唤起|调出来|调出|整开|点开|弄开(?:一下)?|搞开)(?:了|啦)?",
            r"(?:让|把|将)?\s*APP\s*(?:切回|切|放|调|移|弄|回)?(?:到|至)?(?:前台|最前面|最上层)(?:来|去)?",
            r"让\s*APP\s*显示在(?:(?:手机)?屏幕上|前台)",
            r"APP\s*[，,]\s*(?:帮我|给我|麻烦)?\s*(?:打开|启动|进入|进一下|开一下|快开|切过去|切一下)(?:了|啦)?",
        )
    )
    _KNOWN_COMPLEX_ACTION_PATTERN = re.compile(
        r"搜索|搜(?:一下|一把)?|查询|查(?:一下|票)?|看(?:看|一下)?|查看|瞧瞧|"
        r"找|浏览|追番|收藏|打分|筛|排序|排一下|导航|路线|"
        r"发送|发布|发(?:一句|出去|条|个|消息)?|传文件|截图|截个图|存下来|"
        r"播放|随机播|放完|放歌|刷(?:一遍|搞笑)?|刷新|"
        r"标成|标记|退掉|退票|买票|交话费|点单|下单|购买|买|"
        r"叫一辆|叫车|预约|订座|订票|打车|拍|修图|点击|点开|"
        r"(?:^|然后|再)点(?!开)|"
        r"然后\s*(?:打开|启动|切到)|再\s*(?:打开|启动|切到|进)"
    )
    _INCOMPLETE_COMPLEX_PATTERN = re.compile(
        r"^\s*(?:APP\s*)?(?:查路线|查询车次|查航班|发消息|传文件|退票|买票|交话费)\s*[。！？!?]*\s*$"
    )
    _UNSUPPORTED_META_REQUEST_PATTERN = re.compile(
        r"^\s*(?:(?:这次|现在|目前)\s*)?(?:需要|想用|要用)\s*APP\s*"
        r"(?:客户端|应用|软件)?\s*[。！？!?]*\s*$|"
        r"^\s*APP\s*(?:这边)?\s*(?:帮我|请)?\s*(?:处理|办理|安排)\s*"
        r"(?:一下)?\s*[^，,。！？!?；;]{0,12}[。！？!?]*\s*$"
    )
    _HIGH_IMPACT_ACTION_PATTERN = re.compile(
        r"(?:退(?:掉|票|款)|取消\s*(?:订单|行程|预约|订票|车票))|"
        r"(?:购物车[^，,。！？!?；;]{0,16})?"
        r"(?:全部|全都|批量)[^，,。！？!?；;]{0,8}(?:下单|购买|买下)|"
        r"(?:下单|购买|买下|买票|叫车|打车|付款|支付|转账|交话费)"
        r"|(?:呼叫(?:快车|专车|出租车)|确认用车)"
        r"|(?:点|订)一份[^，,。！？!?；;]{0,16}并提交"
        r"|(?:发送|发)(?:给[^，,。！？!?；;]{0,12})?(?:消息|文件|照片|通知|评论|动态|帖子|笔记)"
        r"|(?<!已)(?<!已经)(?<!被)发布(?!过|的|时间|记录|历史|状态)"
        r"[^，,。！？!?；;\n]{0,16}?(?:评论|动态|帖子|笔记|内容)"
        r"|(?:评论|动态|帖子|笔记|内容)[^，,。！？!?；;\n]{0,8}"
        r"(?:发布(?!时间|记录|历史|状态|者)|发出去)"
        r"|(?<!已)(?<!被)(?:删除|删掉|清空|永久移除)"
        r"|(?<!已)(?<!被)(?:发送|转发)(?!过|的|时间|记录|历史|状态)"
        r"|(?:发|寄|传)给|发邮件"
        r"|公开发布(?:出去)?|发成公开(?:笔记|帖子|动态|内容)"
        r"|提交(?=[^，,。！？!?；;]{0,20}(?:订单|购物车))"
        r"|(?:付|转|交)(?=(?:人民币|美元|[￥¥])?\s*"
        r"[0-9零一二两三四五六七八九十百千万点.]+(?:元|块|美元|人民币))"
        r"|(?<!已)(?<!已经)(?<!被)取消(?=[^，,。！？!?；;]{0,20}(?:车票|机票|订单|行程|预约|订票))"
        r"|叫[0-9一二两三四五六七八九十]+辆(?:出租|网约|快|专)?车"
    )
    _NEGATED_ACTION_PREFIX = re.compile(
        r"(?:不用|不要|不做|不进行|别|无需|不需要|不必|禁止|不许|不得|不允许|不能|暂不|不)"
        r"(?:再|继续|开始|进入|进行|点击|选择|执行|发起|实际|做)?"
        r"(?:在[^，,。！？!?；;]{0,12})?\s*$"
    )
    _MIXED_APP_CONTEXT_PATTERN = re.compile(
        r"(?:在|用)?\s*(?:"
        r"APP\s*(?:和|与|及|、)\s*(?!APP)[\w\u4e00-\u9fff-]{2,16}?|"
        r"(?!APP)[\w\u4e00-\u9fff-]{2,16}?\s*(?:和|与|及|、)\s*APP"
        r")\s*(?:里|中|内|上)\s*(?:分别|一起|都)?\s*"
        r"(?:搜索|搜|查询|查|查看|浏览|打开|启动|进入)"
    )
    _UNKNOWN_PEER_TARGET_PATTERN = re.compile(
        r"APP\s*(?:和|与|及|、|以及|跟)\s*(?:"
        r"[A-Za-z][A-Za-z0-9_-]*|"
        r"[\u4e00-\u9fff]{1,10}(?:应用|程序|软件|相机|商城|清单|"
        r"日历|电台|云盘|账簿|票务|播客|扫描仪)"
        r")"
    )
    _ALTERNATIVE_PATTERN = re.compile(r"或者|或是|二选一")
    _EXPLICIT_ONE_OF_PATTERN = re.compile(
        r"任选(?:一个|其一)|任意(?:一个|其一)|其中一个|任一|二选一|"
        r"一个(?:即可|就行|便可)"
    )
    _CONDITION_OR_NEGATED_LAUNCH = re.compile(
        r"如果|假如|要是|除非|是否"
    )
    _NEGATED_LAUNCH_PATTERN = re.compile(
        rf"(?:不要|别|禁止|不准|不得|无需|不用|不许|不必|不)\s*(?:再|继续)?\s*"
        rf"(?:(?:{_NEW_FORWARD_LAUNCH_VERB})\s*APP|"
        rf"(?:把|将|让)?\s*APP\s*(?:{_STATE_LAUNCH_VERB}))|"
        rf"APP\s*(?:现在|先|暂时|暂且)*(?:不要|别|禁止|不准|不得|无需|不用|不许|不必|不)\s*(?:再|继续)?\s*"
        rf"(?:{_STATE_LAUNCH_VERB}|{_NEW_FORWARD_LAUNCH_VERB})|"
        r"APP\s*(?:现在|先|暂时|暂且)*(?:不要|别|禁止|不准|不得|无需|不用)\s*"
        r"(?:打开|启动|进入|切到|唤起|恢复到|唤回)|"
        r"(?:不要|别|禁止|不准|不得|无需|不用)\s*(?:把|将|让)?\s*APP\s*"
        r"(?:恢复到|唤回)|"
        r"(?:不要|别|禁止|不准|不得|无需|不用)\s*"
        r"(?:打开|启动|进入|切到|切换到|切换至|切换|转到|唤起|"
        r"调出|调到前台|放到前台|移到前台)\s*(?:一下)?\s*APP|"
        r"(?:不要|别|禁止|不准|不得|无需|不用)\s*让\s*APP\s*显示|"
        r"(?:不要|别|禁止|不准|不得|无需|不用)\s*"
        r"(?:把|将)?\s*APP\s*(?:切过来|切到|切换到|打开|启动|进入)"
    )
    _BOUNDED_ATOMIC_LAUNCH_PATTERN = re.compile(
        r"(?:"
        r"切换屏幕至\s*APP|"
        r"把屏幕带到\s*APP(?:的初始界面)?|"
        r"将前台应用改为\s*APP|"
        r"将当前界面换到\s*APP|"
        r"让手机转到\s*APP(?:应用)?|"
        r"启动并呈现\s*APP(?:本身)?|"
        r"唤出\s*APP(?:这个应用)?|"
        r"让\s*APP\s*显示到手机前台|"
        r"把\s*APP\s*(?:带到最上层|从桌面叫出来)"
        r")"
    )
    _ATOMIC_BOUNDARY_PATTERN = re.compile(
        r"看到它在前台便停住|停在应用刚出现的位置|"
        r"出现后立即结束|到这里任务就完成|仅此一步即可|"
        r"接下来不用继续|不要再触碰里面的内容|"
        r"不用搜索(?:，|,)\s*也不用打开条目"
    )
    _SAFE_NEGATIVE_TAIL = re.compile(
        r"^(?:(?:暂时|先)?(?:不用|不要|别|无需)|.*(?:先不用|不用|不要|别)).*$"
        r"|^不打开(?:小说|书籍|报道|文章|视频|商品|房源)[。！!]*$"
    )
    _SAFE_SELF_TAIL = re.compile(
        r"^(?:.*我自己.*|剩下(?:的)?我来.*|后面我自己.*|消息我自己.*|我路上自己.*)$"
    )
    _SAFE_PURPOSE_OR_STATE_TAIL = re.compile(
        r"^(?:让我(?:放松|休息|缓一缓)(?:一下|放松)?|"
        r"(?:先)?(?:到|停在)?(?:主页|首页|主页面|主界面|页面)(?:就行|先停住)?|"
        r"到了?(?:主页|首页|主页面|主界面|页面)(?:就行|先停住)?|先停住|维持当前首页|看到起始页就收手)$"
    )
    _SAFE_LOCAL_PREFIX = re.compile(
        r"^(?:小助手|助手|喂|你好|请|麻烦|能不能|能|可不可以|可以|"
        r"帮我|给我|替我|让|把|将|你|先|直接|现在|现在就|马上|只要|只|就|出门前|屏幕|画面|界面|"
        r"不过还是|我要用|我用|我这会儿要用|我想用|辛苦|呃|那个|就是|嗯|对了|"
        r"\s|[，,。！？!?：:])*$"
    )
    _SAFE_SUFFIX_PARTICLES = re.compile(
        r"^(?:就行|就好|即可|便可|便完成|就够了|就结束|就停止|可以吧|好吗|吧|吗|嘛|哈|喽|咯|呢|了|啦|一下|\s|[，,。！？!?；;：:])*$"
    )
    _NATURAL_ATOMIC_BOUNDARY_PATTERN = re.compile(
        r"(?:"
        r"(?:首页|主页|主界面|起始界面|首个可见界面|初始页面|主页面|页面|启动页|应用本身|前台|这一步)"
        r"[^，,。！？!?；;]{0,24}(?:停|停止|收手|结束|即可|就行|就好|可以|为止|保持|留在|不要再操作|不要继续点击)|"
        r"(?:仅|只)?(?:停留|停在|留在|保持)[^，,。！？!?；;]{0,18}"
        r"(?:首页|主页|主界面|起始界面|首个可见界面|初始页面|主页面|页面|启动页|前台)|"
        r"(?:只需要|只要|只做到|只停在|只进入|只到)"
        r"[^，,。！？!?；;]{0,24}(?:首页|主页|主界面|起始界面|首个可见界面|初始页面|主页面|页面|应用本身|前台|这一步)|"
        r"(?:应用)?打开后[^，,。！？!?；;]{0,12}停|"
        r"只做应用切换|"
        r"(?:起来)?(?:即可|就行|就好)"
        r")"
    )
    _NATURAL_EXPLICIT_LAUNCH_PATTERN = re.compile(
        r"(?:打开|启动|进入|进|唤起|唤出|唤醒|开|切到|切换到|切至)"
        r"\s*(?:一下)?\s*APP|"
        r"(?:把|将|让|替我)?\s*APP"
        r"(?:应用|本体)?(?:从后台)?\s*(?:现在)?(?:只需要|仅需|只要|只需)?\s*"
        r"(?:打开|启动|进入|被打开|开到|切到|切来|切回|切过去|"
        r"切出来|调出|调出来|调到|唤起|唤出|唤出来|唤到|带到|带回|放到|"
        r"放回|置于|显示在|显示出来|出现在|亮出|开|恢复到|唤回)(?:到|在|来)?|"
        r"(?:把|将|让|替我)\s*APP[^，,。！？!?；;]{0,12}"
        r"(?:前台|屏幕|最前面|当前窗口|当前界面|眼前)|"
        r"APP\s*的\s*启动|"
        r"APP[^。！？!?；;]{0,24}(?:重新)?(?:开到前台|切过去)|"
        r"APP[^。！？!?；;]{0,20}(?:打开|启动|调出)(?:它|这个应用)|"
        r"(?:看见|看到|让我看见)\s*APP[^，,。！？!?；;]{0,10}"
        r"(?:启动|应用页面|界面)|"
        r"(?:屏幕上\s*)?(?:请)?(?:先)?显示\s*APP|"
        r"APP[^，,。！？!?；;]{0,10}(?:应用)?界面"
    )
    _NATURAL_ATOMIC_LIMIT_PATTERN = re.compile(
        r"只|仅|即可|便可|就可以|就行|就好|为止|到此|"
        r"停|停手|收手|收住|结束|完成|保持|留在|不动|原样|稍后再说|"
        r"我来|我自己|自己操作|由我|交给我|让我接手|其余|后续|后面|"
        r"不用|不要|别|无需|不需|不必|暂不|暂时不|先不|不继续|不打开(?:小说|书籍|报道|文章|视频|商品|房源)(?=$|[，,。！？!?；;])|不(?:搜|查|看|点|播|叫|输入|规划|浏览|筛选|发起|办理|选|做|处理|翻)"
    )
    _NEGATED_UNKNOWN_LAUNCH_PATTERN = re.compile(
        r"(?:不用|不要|别|无需|不需要|不必)\s*(?:再)?\s*"
        r"(?:打开|启动|切到|切换到|进入)\s*"
        r"(?!页面|内容|条目|回答|订单|联系人|会话|消息|文档|商品|视频|聊天|店铺|栏目|频道|任何频道|报道|支付|功能|详情|文件)"
        r"[^\s，,。！？!?；;]{2,}"
    )
    _MISSING_TARGET_PATTERN = re.compile(
        r"^\s*(?:请|帮我|麻烦|能不能|可以)?\s*"
        r"(?:打开|启动|进入|进|开一下|切到|切换|切过去|弄一下|订)\s*"
        r"(?:那个|这个|它|过去|明天的)?\s*(?:好吗|吧|呗|嘛)?\s*[，,。！？!?]*\s*$"
    )
    _VAGUE_TARGET_PATTERN = re.compile(
        r"那个|这个|它|上回|我说的|(?:我|你)(?:刚才|之前|先前)(?:说|提到)的应用|图标|左边|右边|上边|下边|左侧|右侧"
    )
    _READ_ACTION = r"搜索|查询|查找|检索|查看|浏览|预览|阅读|定位|搜|查|找|看|读"
    # Navigation/comparison use the same complete app-slot binding as search,
    # without changing the no-app task-domain fallback.
    _APP_SCOPED_READ_ACTION = _READ_ACTION + r"|导航|比较|估价"
    _READ_APP_CONTEXT = re.compile(
        r"(?:使用(?!(?:手册|指南|说明|教程)(?!软件|应用|程序|客户端|App))|"
        r"借助|通过|(?<!现)在|(?<![找看翻开高低调移切放转位收])到|"
        r"(?<!别)(?<!不要)去|(?<![不应使])用)\s*"
        r"(?P<context>"
        r"观看历史(?:里|中|内|上)?(?=(?:" + _APP_SCOPED_READ_ACTION + r"))|"
        r"[^，,。！？!?；;\n的]{1,40}(?:版|助手|浏览器|工具|应用|软件|程序|客户端|App)"
        r"(?:里|里面|中|内|上)?(?=(?:" + _APP_SCOPED_READ_ACTION + r"))|"
        r"[^，,。！？!?；;\n]+?(?=(?:" + _APP_SCOPED_READ_ACTION + r")))",
        re.IGNORECASE,
    )
    _EXPLICIT_UNKNOWN_NAME_PATTERN = re.compile(r"叫\s*[A-Za-z][A-Za-z0-9_-]*", re.IGNORECASE)
    _UNSAFE_TARGET_PATTERN = re.compile(r"微信|WeChat", re.IGNORECASE)
    _SUPPORTED_PAGE_TARGET = (
        r"(?:(?:我的|我喜欢的|我收藏的)?"
        r"(?:首页|主页|消息列表|订单页|收藏夹|日历页|歌单|播放列表|个人中心|收件箱)|"
        r"(?:地点|商品|商户|订单|内容|视频|笔记)?详情页?)"
    )
    _SUPPORTED_PAGE_NAVIGATION_PATTERN = re.compile(
        rf"(?:打开|启动|切到|切换到)\s*APP\s*(?:的|里|内)?\s*"
        rf"(?:[，,]\s*)?(?:(?:后|然后|再)\s*)?"
        rf"(?:(?:进(?:入)?|打开|切到|跳到|回到|翻到|点开)\s*)?{_SUPPORTED_PAGE_TARGET}|"
        rf"APP\s*(?:的|里|内|中)?\s*(?:打开|进(?:入)?|切到|跳到|回到|翻到|点开)\s*"
        rf"{_SUPPORTED_PAGE_TARGET}"
    )
    _APP_DESCRIPTOR_PATTERN = re.compile(
        r"APP\s*(?:这个应用|那款软件|这个\s*App|那个程序|这个App|App|应用|软件|程序)",
        re.IGNORECASE,
    )

    _ALIASES = tuple(
        sorted(
            (
                (alias, canonical)
                for canonical, aliases in FAST_PATH_APP_ALIASES.items()
                for alias in aliases
            ),
            key=lambda item: (-len(item[0]), item[0].casefold(), item[1]),
        )
    )

    def __init__(self, predictor: Callable[[str], str] | None = None):
        self.predictor = predictor

    def decide(self, text: str) -> FastPathDecision:
        """Combine classifier evidence with the deterministic execution contract."""
        stripped = text.strip()
        if not stripped:
            return FastPathDecision(
                kind=FastPathDecisionKind.UNSUPPORTED_OR_CLARIFY,
                reason="empty_input",
                message="Task is empty. Please provide a valid instruction.",
                classifier_outcome="not_run_empty",
                parser_status=FastPathParserStatus.MISSING_TARGET.value,
                guard_outcome="reject_or_clarify",
            )

        model_text = stripped
        stripped = self._current_request_text(stripped)
        had_background = stripped != model_text
        parsed = self.parse_launch_contract(stripped)
        raw_prediction: str | None = None
        classifier_label: str | None = None
        classifier_outcome: str
        classifier_error: str | None = None

        if self.predictor is None:
            classifier_outcome = "router_unavailable"
        else:
            try:
                raw_prediction = self.predictor(model_text)
                classifier_label = self._normalize_label(raw_prediction)
                classifier_outcome = (
                    f"label_{classifier_label}"
                    if classifier_label is not None
                    else "invalid_label"
                )
            except Exception as exc:
                classifier_outcome = "router_error"
                classifier_error = str(exc)

        common = {
            "app_name": parsed.app_name,
            "route_label": classifier_label,
            "raw_prediction": raw_prediction,
            "classifier_label": classifier_label,
            "classifier_outcome": classifier_outcome,
            "parser_status": parsed.status.value,
            "matched_app_count": parsed.matched_app_count,
            "launch_intent_count": parsed.launch_intent_count,
        }

        # The learned model owns the route choice. The parser/guard may only
        # preserve that choice or make it safer; it must never promote a model
        # Slow/Reject (or an unavailable model) into Direct Fast.
        if classifier_label == "2":
            return FastPathDecision(
                kind=FastPathDecisionKind.UNSUPPORTED_OR_CLARIFY,
                reason="model_reject",
                guard_outcome="reject_or_clarify",
                message="Route model rejected this request or requires clarification.",
                **common,
            )

        if classifier_label is None:
            message = "Route model did not produce a usable decision; request was not executed."
            if classifier_error:
                message = (
                    f"Fast Path router failed ({classifier_error}); request was not executed."
                )
            return FastPathDecision(
                kind=FastPathDecisionKind.UNSUPPORTED_OR_CLARIFY,
                reason=classifier_outcome,
                guard_outcome="model_unavailable_fail_closed",
                message=message,
                **common,
            )

        if self._has_unbound_read_context(
            stripped, self._find_allowed_app_mentions(stripped)
        ) and not (
            self._parse_multiple_app_launch(stripped) == FastPathParserStatus.KNOWN_COMPLEX_TASK
            or self._is_explicit_app_choice(stripped, self._find_allowed_app_mentions(stripped))
        ):
            return FastPathDecision(
                kind=FastPathDecisionKind.UNSUPPORTED_OR_CLARIFY,
                reason="unbound_read_app",
                guard_outcome="reject_or_clarify",
                message="The query has no supported app target.",
                **common,
            )

        if classifier_label == "0":
            guarded_app = self._guarded_label0_app(stripped, parsed)
            if guarded_app is not None and not had_background:
                common["app_name"] = guarded_app
                return FastPathDecision(
                    kind=FastPathDecisionKind.FAST_LAUNCH,
                    reason="model_fast_guard_confirmed",
                    guard_outcome="allow_fast",
                    message=f"Fast Path v1: local launch for {guarded_app}.",
                    **common,
                )

        if (
            classifier_label == "0"
            and (
                parsed.status in self._HARD_GUARD_REJECT_STATUSES
                or self._has_negated_launch(
                    stripped, self._find_allowed_app_mentions(stripped)
                )
                or self._unsupported_contract_needs_hard_reject(
                    stripped, parsed
                )
            )
        ) or (
            classifier_label == "1"
            and self._slow_model_needs_hard_reject(stripped, parsed)
        ):
            return FastPathDecision(
                kind=FastPathDecisionKind.UNSUPPORTED_OR_CLARIFY,
                reason=parsed.status.value,
                guard_outcome="reject_or_clarify",
                message="This request is unsafe, ambiguous, or has no supported target.",
                **common,
            )

        if classifier_label == "1":
            return FastPathDecision(
                kind=FastPathDecisionKind.FALLBACK_SLOW,
                reason="model_slow",
                guard_outcome="allow_slow",
                message="Route model selected Slow Path.",
                **common,
            )

        # Only label 0 reaches this point. If its launch contract is not safe
        # and executable, downgrade to Slow rather than letting the parser
        # invent a Fast decision or rejecting an otherwise recoverable task.
        return FastPathDecision(
            kind=FastPathDecisionKind.FALLBACK_SLOW,
            reason="guard_veto_fast",
            guard_outcome="downgrade_fast_to_slow",
            message="Fast prediction did not satisfy the direct-execution contract; using Slow Path.",
            **common,
        )

    @classmethod
    def _slow_model_needs_hard_reject(
        cls,
        text: str,
        parsed: FastPathParseResult,
    ) -> bool:
        """Keep Slow model decisions unless a narrow hard-risk signal is present."""
        if parsed.status in {
            FastPathParserStatus.UNSAFE_TARGET,
            FastPathParserStatus.MISSING_TARGET,
        }:
            return True
        if cls._unsupported_contract_needs_hard_reject(text, parsed):
            return True
        mentions = cls._find_allowed_app_mentions(text)
        if cls._parse_multiple_app_launch(text) in {
            FastPathParserStatus.UNKNOWN_TARGET, FastPathParserStatus.AMBIGUOUS_TARGET,
        }:
            return True
        return parsed.status == FastPathParserStatus.UNKNOWN_TARGET and (
            not mentions or cls._has_unbound_launch_target(text, mentions, allow_page_targets=True)
        )

    @classmethod
    def _unsupported_contract_needs_hard_reject(
        cls,
        text: str,
        parsed: FastPathParseResult,
    ) -> bool:
        """Reject only unsupported clauses that have no safe executable intent."""
        if parsed.status != FastPathParserStatus.UNSUPPORTED_CONTRACT:
            return False
        mentions = cls._find_allowed_app_mentions(text)
        if not mentions:
            return bool(
                cls._EXPLICIT_UNKNOWN_NAME_PATTERN.search(text)
                or re.search(
                    r"(?<![A-Za-z0-9_])[A-Z][A-Za-z0-9_-]{2,}", text
                )
            )
        if (
            cls._CONDITION_OR_NEGATED_LAUNCH.search(text)
            or cls._DEFERRED_OR_META_LAUNCH_PATTERN.search(text)
            or cls._has_deferred_launch(text, mentions)
            or cls._has_negated_launch(text, mentions)
        ):
            return True
        if len(mentions) != 1:
            return True
        normalized = cls._normalize_contract_text(text, mentions[0])
        return bool(cls._UNSUPPORTED_META_REQUEST_PATTERN.fullmatch(normalized))

    @classmethod
    def _guarded_label0_app(
        cls,
        text: str,
        parsed: FastPathParseResult,
    ) -> str | None:
        """Confirm only the narrow conditions needed for direct app launch."""
        if parsed.status in {
            FastPathParserStatus.KNOWN_COMPLEX_TASK,
            FastPathParserStatus.UNSAFE_TARGET,
            FastPathParserStatus.AMBIGUOUS_TARGET,
        }:
            return None
        mentions = cls._find_allowed_app_mentions(text)
        if len(mentions) != 1:
            return None
        if cls._UNSAFE_TARGET_PATTERN.search(text):
            return None
        if cls._has_high_impact_action(text, mentions):
            return None
        if cls._has_mixed_app_context(text, mentions[0]):
            return None
        if cls._ALTERNATIVE_PATTERN.search(text):
            return None
        if cls._CONDITION_OR_NEGATED_LAUNCH.search(text):
            return None
        if cls._DEFERRED_OR_META_LAUNCH_PATTERN.search(text):
            return None
        if cls._has_negated_launch(text, mentions):
            return None
        if parsed.status == FastPathParserStatus.VALID_ATOMIC_LAUNCH:
            return mentions[0][2]
        return None

    @staticmethod
    def _normalize_label(route_label: str | None) -> str | None:
        if route_label is None:
            return None

        raw = str(route_label).strip()
        if raw in {"0", "1", "2"}:
            return raw

        match = re.search(r"\b([012])\b", raw)
        if match:
            return match.group(1)
        return None

    @classmethod
    def _current_request_text(cls, text: str) -> str:
        """Separate completed background sentences from an explicit new launch."""
        for boundary in re.finditer(r"[。！？!?]", text):
            background, current = text[:boundary.end()], text[boundary.end():].strip()
            if not current or not cls._find_allowed_app_mentions(background):
                continue
            if not re.search(r"收到|收过|给过|寄来|寄进|寄到|寄了|寄过|传来|传过|发来|发给|发过|发送过|寄给|传给|补发了|分享过|看过|交给|有一封", background):
                continue
            # Never discard an earlier command or an app launch as history.
            if (cls._FORWARD_TARGET_LAUNCH_CUE_PATTERN.search(background)
                or re.search(cls._STATE_LAUNCH_VERB + r"|回到前台|显示在前台", background)
                or re.search(
                r"请|帮我|替我|现在|马上|立即|然后|接着|随后|(?:再|要)(?:发送|转发|删除|付款|支付)", background
            )):
                continue
            background_mentions = cls._find_allowed_app_mentions(background)
            # Reuse every supported launch grammar, including screen/state
            # orders without a forward verb. A receipt in the same sentence
            # cannot erase a launch that the ordinary parser already binds.
            if any(
                cls._find_launch_contracts(background, mention)
                or cls._BOUNDED_ATOMIC_LAUNCH_PATTERN.search(
                    cls._normalize_contract_text(background, mention)
                )
                or cls._NATURAL_EXPLICIT_LAUNCH_PATTERN.search(
                    cls._normalize_contract_text(background, mention)
                )
                for mention in background_mentions
            ):
                continue
            if cls._has_high_impact_action(background, background_mentions) or cls._has_unbound_read_context(
                background, background_mentions
            ):
                continue
            if (
                cls._LAUNCH_CUE_PATTERN.search(current)
                or re.search(cls._STATE_LAUNCH_VERB + r"|显示在前台|回到前台", current)
            ):
                return current
        return text

    @classmethod
    def parse_launch_contract(cls, text: str) -> FastPathParseResult:
        """Parse one request without consulting the learned classifier."""
        stripped = cls._current_request_text(text.strip())
        if not stripped:
            return FastPathParseResult(FastPathParserStatus.MISSING_TARGET)

        mentions = cls._find_allowed_app_mentions(stripped)
        unsafe_mentions = tuple(cls._UNSAFE_TARGET_PATTERN.finditer(stripped))
        if unsafe_mentions:
            return FastPathParseResult(
                FastPathParserStatus.UNSAFE_TARGET,
                matched_app_count=max(len(mentions), len(unsafe_mentions)),
            )

        if cls._has_high_impact_action(stripped, mentions):
            return FastPathParseResult(
                FastPathParserStatus.UNSAFE_TARGET,
                matched_app_count=len(mentions),
            )

        multiple_launch = cls._parse_multiple_app_launch(stripped)
        if (
            multiple_launch == FastPathParserStatus.UNKNOWN_TARGET
            and len(mentions) == 1
            and re.match(r"助手|轻聊版|极速版|专业版|测试版|浏览器|工具", stripped[mentions[0][1]:])
        ):
            return FastPathParseResult(multiple_launch, matched_app_count=len(mentions))

        # A harmless tail must not turn a deferred launch into an immediate
        # executable task through the complex-action parser branch below.
        if cls._has_negated_launch(
            stripped, mentions
        ) or cls._has_deferred_launch(stripped, mentions):
            return FastPathParseResult(
                FastPathParserStatus.UNSUPPORTED_CONTRACT,
                app_name=mentions[0][2] if len(mentions) == 1 else None,
                matched_app_count=len(mentions),
            )

        if len(mentions) == 1 and cls._has_mixed_app_context(stripped, mentions[0]):
            return FastPathParseResult(
                FastPathParserStatus.UNKNOWN_TARGET,
                matched_app_count=1,
            )

        if len(mentions) == 1 and cls._has_unknown_peer_target(
            stripped, mentions[0]
        ):
            return FastPathParseResult(
                FastPathParserStatus.UNKNOWN_TARGET,
                matched_app_count=1,
            )

        # A complete one-of-two reference with an unbound candidate is not a
        # single-app launch, even if the other candidate is whitelisted.
        if cls._has_unknown_referential_choice(stripped):
            return FastPathParseResult(
                FastPathParserStatus.UNKNOWN_TARGET,
                matched_app_count=len(mentions),
            )

        if len(mentions) == 1:
            bounded_text = cls._normalize_contract_text(stripped, mentions[0])
            bare_tail = re.fullmatch(r"APP\s*([，,。！？!?；;：:].*)?", bounded_text)
            if bare_tail and (
                not bare_tail[1] or all(
                    not clause.strip() or re.fullmatch(
                        r"\s*(?:不用|无需|不要|别)(?:再|继续)?"
                        r"(?:处理|查看|阅读|搜索|浏览|选择|播放|输入)"
                        r"(?:任何|里面的|应用内的)?"
                        r"(?:邮件|消息|文章|内容|文件|文档|商品|视频|歌曲|信息|问题|数据)(?:了)?\s*",
                        clause,
                    )
                    for clause in re.split(r"[，,。！？!?；;：:]", bare_tail[1])
                )
            ):
                return FastPathParseResult(
                    FastPathParserStatus.VALID_ATOMIC_LAUNCH,
                    app_name=mentions[0][2],
                    matched_app_count=1,
                    launch_intent_count=1,
                )
            if (
                cls._BOUNDED_ATOMIC_LAUNCH_PATTERN.search(bounded_text)
                and cls._ATOMIC_BOUNDARY_PATTERN.search(bounded_text)
            ):
                return FastPathParseResult(
                    FastPathParserStatus.VALID_ATOMIC_LAUNCH,
                    app_name=mentions[0][2],
                    matched_app_count=1,
                    launch_intent_count=1,
                )
            if cls._is_natural_atomic_launch(
                stripped, mentions[0]
            ) or cls._is_general_natural_atomic_launch(stripped, mentions[0]):
                return FastPathParseResult(
                    FastPathParserStatus.VALID_ATOMIC_LAUNCH,
                    app_name=mentions[0][2],
                    matched_app_count=1,
                    launch_intent_count=1,
                )

        if not mentions and (
            cls._MISSING_TARGET_PATTERN.fullmatch(stripped)
            or cls._VAGUE_TARGET_PATTERN.search(stripped)
            or re.search(r"打开\s*(?:但|可是)?\s*(?:不要|别)\s*打开", stripped)
        ):
            return FastPathParseResult(FastPathParserStatus.MISSING_TARGET)

        if cls._is_explicit_app_choice(stripped, mentions):
            return FastPathParseResult(
                FastPathParserStatus.KNOWN_COMPLEX_TASK,
                matched_app_count=len(mentions),
            )

        if multiple_launch is not None:
            return FastPathParseResult(
                multiple_launch,
                matched_app_count=len(mentions),
            )

        if cls._has_unbound_launch_target(stripped, mentions):
            return FastPathParseResult(
                FastPathParserStatus.UNKNOWN_TARGET,
                matched_app_count=len(mentions),
            )

        if len(mentions) > 1:
            if (
                re.search(r"还是|或者|或是|二选一", stripped)
                and not cls._EXPLICIT_ONE_OF_PATTERN.search(stripped)
            ):
                status = (
                    FastPathParserStatus.KNOWN_COMPLEX_TASK
                    if cls._is_explicit_app_choice(stripped, mentions)
                    else FastPathParserStatus.AMBIGUOUS_TARGET
                )
            else:
                status = FastPathParserStatus.KNOWN_COMPLEX_TASK
            return FastPathParseResult(
                status,
                matched_app_count=len(mentions),
                launch_intent_count=len(cls._find_launch_contracts(stripped, mentions[0])),
            )

        if not mentions:
            if cls._EXPLICIT_UNKNOWN_NAME_PATTERN.search(stripped):
                return FastPathParseResult(FastPathParserStatus.UNKNOWN_TARGET)
            if (
                cls._MISSING_TARGET_PATTERN.fullmatch(stripped)
                or cls._VAGUE_TARGET_PATTERN.search(stripped)
                or re.search(r"打开\s*(?:但|可是)?\s*(?:不要|别)\s*打开", stripped)
            ):
                return FastPathParseResult(FastPathParserStatus.MISSING_TARGET)
            if cls._LAUNCH_CUE_PATTERN.search(stripped):
                return FastPathParseResult(FastPathParserStatus.UNKNOWN_TARGET)
            return FastPathParseResult(FastPathParserStatus.UNSUPPORTED_CONTRACT)

        mention = mentions[0]
        app_name = mention[2]
        contract_text = cls._normalize_contract_text(stripped, mention)
        contracts = cls._find_launch_contracts(stripped, mention)
        if len(contracts) > 1:
            return FastPathParseResult(
                FastPathParserStatus.KNOWN_COMPLEX_TASK,
                app_name=app_name,
                matched_app_count=1,
                launch_intent_count=len(contracts),
            )

        if not contracts:
            if cls._SUPPORTED_PAGE_NAVIGATION_PATTERN.search(contract_text):
                return FastPathParseResult(
                    FastPathParserStatus.KNOWN_COMPLEX_TASK,
                    app_name=app_name,
                    matched_app_count=1,
                )
            status = cls._classify_complex_clause(contract_text)
            return FastPathParseResult(
                status or FastPathParserStatus.UNSUPPORTED_CONTRACT,
                app_name=app_name,
                matched_app_count=1,
            )

        contract_text, contract_match = contracts[0]
        prefix = contract_text[: contract_match.start()]
        suffix = contract_text[contract_match.end() :]
        prefix_safe = cls._prefix_is_safe(prefix)
        suffix_safe = cls._suffix_is_safe(suffix)
        if cls._CONDITION_OR_NEGATED_LAUNCH.search(contract_text):
            return FastPathParseResult(
                FastPathParserStatus.UNSUPPORTED_CONTRACT,
                app_name=app_name,
                matched_app_count=1,
                launch_intent_count=1,
            )

        if cls._SUPPORTED_PAGE_NAVIGATION_PATTERN.search(contract_text):
            return FastPathParseResult(
                FastPathParserStatus.KNOWN_COMPLEX_TASK,
                app_name=app_name,
                matched_app_count=1,
                launch_intent_count=1,
            )

        prefix_complex = cls._classify_complex_clause(prefix)
        suffix_complex = cls._classify_complex_clause(suffix)
        if prefix_complex == FastPathParserStatus.UNSUPPORTED_CONTRACT or (
            not suffix_safe
            and suffix_complex == FastPathParserStatus.UNSUPPORTED_CONTRACT
        ):
            return FastPathParseResult(
                FastPathParserStatus.UNSUPPORTED_CONTRACT,
                app_name=app_name,
                matched_app_count=1,
                launch_intent_count=1,
            )
        if prefix_complex == FastPathParserStatus.KNOWN_COMPLEX_TASK or (
            not suffix_safe
            and suffix_complex == FastPathParserStatus.KNOWN_COMPLEX_TASK
        ):
            return FastPathParseResult(
                FastPathParserStatus.KNOWN_COMPLEX_TASK,
                app_name=app_name,
                matched_app_count=1,
                launch_intent_count=1,
            )

        if not prefix_safe or not suffix_safe:
            return FastPathParseResult(
                FastPathParserStatus.UNSUPPORTED_CONTRACT,
                app_name=app_name,
                matched_app_count=1,
                launch_intent_count=1,
            )

        return FastPathParseResult(
            FastPathParserStatus.VALID_ATOMIC_LAUNCH,
            app_name=app_name,
            matched_app_count=1,
            launch_intent_count=1,
        )

    @classmethod
    def _has_external_natural_launch(cls, clause: str) -> bool:
        """Do not let an atomic stop tail hide a second, unbound app target."""
        for cue in cls._FORWARD_TARGET_LAUNCH_CUE_PATTERN.finditer(clause):
            target = re.sub(r"^\s*(?:一下)?\s*", "", clause[cue.end():])
            if target and not cls._NATURAL_LAUNCH_REFERENCE.match(target):
                return True
        return False

    @classmethod
    def _is_natural_atomic_launch(
        cls,
        text: str,
        mention: tuple[int, int, str],
    ) -> bool:
        """Accept a launch followed only by a natural stop/hand-off boundary."""
        contracts = cls._find_launch_contracts(text, mention)
        if len(contracts) != 1:
            return False
        contract_text, contract_match = contracts[0]
        prefix = contract_text[: contract_match.start()]
        suffix = contract_text[contract_match.end() :]
        launch_before_app = re.sub(
            rf"(?:{cls._DIRECT_LAUNCH_CUE_PATTERN.pattern})\s*$", "", prefix
        )
        if not cls._prefix_is_safe(prefix) and not cls._prefix_is_safe(
            launch_before_app
        ):
            return False
        if cls._SUPPORTED_PAGE_NAVIGATION_PATTERN.search(contract_text):
            return False
        if cls._NEGATED_UNKNOWN_LAUNCH_PATTERN.search(suffix):
            return False
        if not cls._NATURAL_ATOMIC_BOUNDARY_PATTERN.search(suffix):
            return False

        for clause in cls._ACTION_CLAUSE_SEPARATOR.split(suffix):
            if cls._LOCAL_CONTENT_TAIL.fullmatch(clause.strip()):
                continue
            if cls._has_external_natural_launch(clause):
                return False
            for action in cls._KNOWN_COMPLEX_ACTION_PATTERN.finditer(clause):
                local_prefix = clause[:action.start()]
                if (
                    action.group() == "看"
                    and re.match(
                        r"看(?:到|见)(?:(?:它|应用)的)?"
                        r"(?:首页|主页|主界面|起始界面|起始页|首个可见界面|页面)",
                        clause[action.start():],
                    )
                ):
                    continue
                if not (
                    cls._NEGATED_ACTION_PREFIX.search(local_prefix)
                    or cls._NEGATED_CONTENT_CONTINUATION.search(local_prefix)
                ):
                    return False
        return True

    @classmethod
    def _is_general_natural_atomic_launch(
        cls,
        text: str,
        mention: tuple[int, int, str],
    ) -> bool:
        """Recognize one explicitly bounded Launch across natural word orders."""
        contract_text = cls._normalize_contract_text(text, mention)
        launch = cls._NATURAL_EXPLICIT_LAUNCH_PATTERN.search(contract_text)
        if launch is None or not cls._NATURAL_ATOMIC_LIMIT_PATTERN.search(
            contract_text
        ):
            return False
        if cls._SUPPORTED_PAGE_NAVIGATION_PATTERN.search(contract_text):
            return False
        if cls._NEGATED_UNKNOWN_LAUNCH_PATTERN.search(contract_text):
            return False

        action_text = (
            f"{contract_text[:launch.start()]}"
            f"{' ' * (launch.end() - launch.start())}"
            f"{contract_text[launch.end():]}"
        )
        for clause in cls._ACTION_CLAUSE_SEPARATOR.split(action_text):
            if cls._LOCAL_CONTENT_TAIL.fullmatch(clause.strip()):
                continue
            if cls._has_external_natural_launch(clause):
                return False
            for action in cls._KNOWN_COMPLEX_ACTION_PATTERN.finditer(clause):
                local_prefix = clause[:action.start()]
                if (
                    action.group() == "看"
                    and re.match(
                        r"看(?:到|见)(?:(?:它|应用)的)?"
                        r"(?:启动后的页面|首页|首屏|主页|主界面|起始界面|起始页|页面|界面)",
                        clause[action.start():],
                    )
                ):
                    continue
                if re.search(r"我自己|由我|我来|交给我", local_prefix):
                    continue
                if (
                    cls._NEGATED_ACTION_PREFIX.search(local_prefix)
                    or cls._NEGATED_CONTENT_CONTINUATION.search(local_prefix)
                ):
                    continue
                return False
        return True

    @classmethod
    def _action_is_negated(cls, text: str, start: int, end: int) -> bool:
        """Bind a prohibition to one predicate or an explicit coordinated list."""
        before = cls._ACTION_CLAUSE_SEPARATOR.split(text[:start])[-1]
        after = cls._ACTION_CLAUSE_SEPARATOR.split(text[end:])[0]
        predicate = r"(?:搜索|查询|查看|打开|进入|点开|发送|转发|购买|下单|付款|支付|删除|删掉|清空|叫车|提交|取消)"
        # A shared prohibition only crosses complete predicate names joined
        # by a coordinator, never an arbitrary object or a new positive verb.
        prefix = re.sub(rf"(?:{predicate}\s*(?:或|和|与|及|、)\s*)+$", "", before)
        prefix = re.sub(
            r"(?:把|将|向|给|替|为)(?:(?!但|再|然后|必须|立即|要|打开|启动|购买|付款|支付|发送|转发|删除)[\w\u4e00-\u9fff]){1,16}\s*$",
            "", prefix,
        )
        negated = re.search(
            cls._NEGATED_ACTION_PREFIX.pattern + r"|(?:不是|并非)(?:要|让|叫)(?:你)?\s*$",
            prefix,
        )
        if negated:
            outer_prefix = prefix[:negated.start()]
            # Use the same prohibition vocabulary at both nesting levels;
            # e.g. 不必不发送 must not become a harmless single negation.
            if not (
                cls._NEGATED_ACTION_PREFIX.search(outer_prefix)
                or re.search(
                    r"(?:没|并非|不是|(?:不是|并非)(?:要|让|叫)(?:你)?)\s*$",
                    outer_prefix,
                )
            ):
                return True
        if re.match(r"的(?:事|事情)(?:先|暂时)?(?:不做|不执行|不用执行)(?=$|[\s。！!])", after):
            return True
        # Postposed negation covers this predicate and only its named peers.
        return bool(re.match(
            rf"(?:\s*(?:和|或|与|及|、)\s*{predicate})*\s*"
            r"(?:都|均)?(?:不用|无需|不必|不要)(?:再|继续)?(?:执行|进行|做)(?:了)?(?=$|[\s。！!])",
            after,
        ))

    @classmethod
    def _action_is_read_content(cls, text: str, start: int, end: int) -> bool:
        """A title/folder is a query object; later actions are checked separately."""
        before = cls._ACTION_CLAUSE_SEPARATOR.split(text[:start])[-1]
        after = cls._ACTION_CLAUSE_SEPARATOR.split(text[end:])[0]
        if re.search(r"已(?:经)?$", before) and re.match(r"(?:文件)?夹", after):
            return True
        # Bind only this completed mail predicate, not an entire historical
        # sentence. Subsequent commands still pass through the impact scan.
        historical_noun = (
            r"(?:(?!现在|当前|请|必须|需要|不得|不允许|再|发送|转发|"
            r"删除|删掉|清空|支付|付款|下单|打开|启动)[^，,。！？!?；;\n])"
        )
        if (
            text[start:end] in {"发送", "转发"}
            and re.fullmatch(
                r"\s*(?:我|我们|他|她)曾(?:经)?(?:用|通过)邮件(?:把|将)"
                + historical_noun + r"{1,32}", before
            )
            and re.fullmatch(r"给" + historical_noun + r"{1,24}\s*", after)
        ):
            return True
        past = r"(?:昨天|前天|上周|上次|之前|先前|早先|刚才)"
        # A timestamped completed fact is not a present command. Require
        # its completed suffix and reject imperative/current-action markers.
        if re.match(past + r"(?!的|那|这).+", before) and after.endswith("了") and not re.search(
            r"请|帮我|替我|麻烦|现在|当前|马上|立刻|需要|然后|随后|再", before
        ):
            return True
        if re.search(past + r"\s*$", before) and re.search(cls._READ_ACTION, before) and re.match(
            r"[^，,。！？!?；;]{0,16}的(?:邮件|文件|文档|消息|附件|照片|内容)", after
        ):
            return True
        if not re.search(r"查找|搜索|查询|查看|检索|查|看|读|主题为|标题|名为|叫", before):
            return False
        if re.search(r"(?:搜索|查找|查询|查看|检索|查|看)\s*$", before) and re.match(
            r"(?:(?!再|然后|随后|并|发送|购买|删除)[^，,。！？!?；;]){0,24}的(?:方法|步骤|教程|规则|政策|条件)(?=$|[，,。！？!?；;\s])",
            after,
        ):
            return True
        # Bind the title at the query object, not after a completed read.
        # "查看文件后发送规范" requests a send, not a title lookup.
        title_prefix = re.search(
            r"(?:查找|搜索|查询|查看|检索|查|看|读|主题为|"
            r"标题(?:为|带有|包含|含)|名为|叫)[《“\"]?\s*$", before
        )
        described_object = before.endswith("的") and re.search(
            r"查找|搜索|查询|查看|检索|查|看|读", before
        )
        return bool((title_prefix or described_object) and re.match(
            r"(?:操作)?(?:指南|规范|说明|教程|流程|须知)[》”\"]?"
            r"(?:的|[，,。！？!?；;\s]|$)", after
        ))

    @classmethod
    def _has_high_impact_action(
        cls,
        text: str,
        mentions: tuple[tuple[int, int, str], ...],
    ) -> bool:
        """Reject destructive or bulk purchase actions before Slow routing."""
        action_text = text
        for start, end, _canonical in reversed(mentions):
            action_text = f"{action_text[:start]}{' ' * (end - start)}{action_text[end:]}"
        for match in cls._HIGH_IMPACT_ACTION_PATTERN.finditer(action_text):
            if not (
                cls._action_is_negated(action_text, match.start(), match.end())
                or cls._action_is_read_content(action_text, match.start(), match.end())
                or re.search(r"待\s*$", action_text[max(0, match.start() - 20):match.start()])
            ):
                return True
        return False

    @classmethod
    def _bound_launch_matches(cls, text, mention):
        """Share the existing launch grammar with vetoes and temporal scope."""
        normalized = cls._normalize_contract_text(text, mention)
        matches = [match for _, match in cls._find_launch_contracts(text, mention)]
        for pattern in (
            cls._BOUNDED_ATOMIC_LAUNCH_PATTERN,
            cls._NATURAL_EXPLICIT_LAUNCH_PATTERN,
        ):
            matches.extend(pattern.finditer(normalized))
        return normalized, matches

    @classmethod
    def _has_deferred_launch(cls, text, mentions) -> bool:
        if cls._DEFERRED_LAUNCH_PATTERN.search(text):
            return True
        future = re.compile(
            r"(?:明日|明天|后天)(?:上午|下午|晚上|早上)?(?:再|才)|"
            r"[0-9零一二两三四五六七八九十百]+(?:分钟|小时|天)"
            r"(?:以后|之后|后)(?:再|才)|"
            r"(?:等我|到家|下班|放学|回家)[^，,。！？!?；;]{0,12}"
            r"(?:以后|之后|后)(?:再|才)"
        )
        for mention in mentions:
            normalized, matches = cls._bound_launch_matches(text, mention)
            for match in matches:
                prefix = cls._ACTION_CLAUSE_SEPARATOR.split(normalized[:match.start()])[-1]
                if any(not prefix[item.end():].strip() for item in future.finditer(prefix)):
                    return True
        return False

    @classmethod
    def _has_negated_launch(cls, text, mentions) -> bool:
        """Bind prohibitions and a same-target withdrawal to supported launches."""
        # A fully bound positive sequence also disambiguates 分别打开 from
        # the prohibition 别打开; unparsed/negative continuations keep vetoes.
        if cls._parse_multiple_app_launch(text) == FastPathParserStatus.KNOWN_COMPLEX_TASK:
            return False
        for mention in mentions:
            normalized, matches = cls._bound_launch_matches(text, mention)
            if cls._NEGATED_LAUNCH_PATTERN.search(normalized) or re.search(
                r"APP\s*(?:现在|先|暂时)*(?:别|不要|不)(?:再)?(?:开|进)"
                r"(?=$|[\s，,。！？!?；;])", normalized
            ):
                return True
            for match in matches:
                if cls._action_is_negated(normalized, match.start(), match.end()):
                    return True
                if re.fullmatch(
                    r"\s*(?:还是)?算了(?:先)?(?:别|不要|不用)(?:再)?"
                    r"(?:开|打开|启动|进入)[。！!\s]*", normalized[match.end():]
                ):
                    return True
        return False

    @classmethod
    def _has_mixed_app_context(
        cls,
        text: str,
        mention: tuple[int, int, str],
    ) -> bool:
        """Reject a known app coordinated with an unrecognized peer target."""
        contract_text = cls._normalize_contract_text(text, mention)
        return bool(cls._MIXED_APP_CONTEXT_PATTERN.search(contract_text))

    @classmethod
    def _has_unbound_read_context(cls, text: str, mentions) -> bool:
        """Bind an explicit read/query app slot, not names inside its object."""
        if not mentions:
            return bool(re.search(cls._READ_ACTION, text))
        # Mask complete known spans first: 找 in 贝壳找房 is part of the app,
        # not a query verb. The marker itself cannot match a read action.
        marker = "\0"
        while marker in text:
            marker += "\0"
        masked = text
        for start, end, _ in reversed(mentions):
            masked = masked[:start] + marker + masked[end:]
        preceding_app = False
        # Bind the first entry in each action clause. Do not reinterpret
        # '使用/在/用' inside its later search object as a second app entry.
        clauses = cls._ACTION_CLAUSE_SEPARATOR.split(masked)
        for index, clause in enumerate(clauses):
            match = cls._READ_APP_CONTEXT.search(clause)
            prior_app = preceding_app or bool(match and marker in clause[:match.start()])
            preceding_app = preceding_app or marker in clause
            if match is None:
                continue
            # A completed read followed by 后用/后在 is a new action entry,
            # unlike '查询使用某应用的方法' inside the query object.
            completed = re.match(
                r"(?:查|看|读|浏览|搜索|查询|查看|检索|阅读)完[^，,。！？!?；;]{0,60}?后"
                r"(?=(?:使用|借助|通过|在|到|去|用))", clause[match.end():]
            )
            if completed:
                boundary = match.end() + completed.end()
                clauses.insert(index + 1, clause[boundary:])
                clause = clause[:boundary]
            context = match["context"].strip()
            # A query key requires an app in preceding context. An app name
            # only inside the later search object cannot authorize this slot.
            if prior_app and (
                re.fullmatch(r"(?:文件名|标题|关键词|关键字|日期)(?:来)?", context)
                or re.fullmatch(cls._SUPPORTED_PAGE_TARGET + r"(?:里|中|内|上)?", context)
                or re.fullmatch(
                    r"(?:播放|观看|浏览|搜索|聊天|收件|追剧|收藏|订单|文件)"
                    r"(?:历史|记录|列表|夹|箱|结果)(?:里|中|内|上)?", context
                )
            ):
                continue
            context = re.sub(r"^(?:当前)?白名单(?:内|中|里)的\s*", "", context)
            if not context.startswith(marker):
                return True
            # A peer immediately following the app must also bind. Do not
            # interpret a later '文件名和日期' search modifier as another app.
            if re.match(r"\s*(?:或者|或是|或|和|与|及|、|跟)", context[len(marker):]):
                if not re.fullmatch(
                    re.escape(marker) + r"(?:\s*(?:或者|或是|或|和|与|及|、|跟)\s*"
                    + re.escape(marker) + r")+\s*"
                    r"(?:(?:中|里|内|上|的)|(?:任选|任意选|选择|挑选)(?:一个|其一)|"
                    r"其中一个|分别|各自|各|一起|都|来|进行|\s)*",
                    context,
                ):
                    return True
            else:
                suffix = re.sub(r"\s+", "", context[len(marker):])
                # An attached product name/version is not an alias. Keep
                # ordinary app descriptors and in-app locations unchanged.
                descriptor = r"(?:应用|软件|客户端|程序|App)(?:里|里面|中|内|上)?"
                product_suffix = (
                    r"[\w\u4e00-\u9fff]{0,20}(?:版|助手|浏览器|工具|应用|软件|程序|客户端|App)"
                    r"(?:里|里面|中|内|上)?"
                )
                if re.fullmatch(product_suffix, suffix, re.IGNORECASE) and not re.fullmatch(
                    descriptor, suffix, re.IGNORECASE
                ):
                    return True
        return False

    @classmethod
    def _has_unknown_peer_target(
        cls,
        text: str,
        mention: tuple[int, int, str],
    ) -> bool:
        """Detect an app-looking coordinated target outside the whitelist."""
        contract_text = cls._normalize_contract_text(text, mention)
        return bool(cls._UNKNOWN_PEER_TARGET_PATTERN.search(contract_text))

    @classmethod
    def _has_unknown_referential_choice(cls, text: str) -> bool:
        """Bind both candidate names even under a negative/deferred wrapper."""
        target = r"[^，,；;。！？!?\n]{1,40}?"
        for pattern in (
            rf"能用(?P<first>{target})也能用(?P<second>{target})[，,；;]",
            rf"(?:打开|开)(?P<first>{target})还是(?:打开|开)(?P<second>{target})都可以[，,]",
        ):
            for match in re.finditer(pattern, text):
                for candidate in (match["first"].strip(), match["second"].strip()):
                    bound = cls._find_allowed_app_mentions(candidate)
                    if len(bound) != 1 or bound[0][:2] != (0, len(candidate)):
                        return True
        return False

    @classmethod
    def _is_explicit_app_choice(cls, text, mentions) -> bool:
        """Bind selection to exactly two named apps, not a later article/topic."""
        if len(mentions) != 2 or mentions[0][2] == mentions[1][2]:
            return False
        normalized = text
        for start, end, _ in reversed(mentions):
            normalized = normalized[:start] + "APP" + normalized[end:]
        launch = r"(?:打开|启动|进入|唤起|叫到前台|切到前台)"
        selector = r"(?:任选(?:一个|其一|一款)|任取其一|任意一个|(?:其中|中的)一个|二选一|挑一个|选一个|任一|一个(?:即可|就行|就够))"
        pair = re.search(
            r"APP\s*(?:(?:或者|或是|或|和|与|跟|、)|[，,]\s*也可以\s*"
            + launch + r")\s*APP", normalized,
        )
        if pair and re.search(selector, normalized[pair.end():]):
            leading = normalized[:pair.start()]
            following = normalized[pair.end():]
            # Every token around the complete two-app pair must belong to
            # candidate declaration, singular selection or a bounded stop.
            lead = (
                r"(?:(?:这次|此次|现在|只需|只要|请|帮我|从|把|将|可以|要|待|"
                r"的|候选|应用|只有|是)|" + launch + r"|\s)*"
            )
            singular = (
                r"(?:(?:中|里面|里|的|你|由你|两者|这两个|哪一个|都行|只|"
                r"一个|其一|即可|就行|就好|就够|够了|便完成|完成|后停住)|"
                r"是两(?:个|款)(?:明确)?候选(?:应用)?|"
                + selector + r"|" + launch + r"|[，,。；;\s])*"
            )
            stop = (
                r"(?:不要求(?:两款|两者|两个(?:应用)?)(?:都|全部)(?:开|打开|启动)|"
                r"另一个(?:先)?(?:不动|不用开|无需启动|不用启动)|"
                r"(?:先|暂时)?(?:不|不要|不用)(?:打开|进入|播放|选)"
                r"(?:文档|内容|视频|歌曲|歌)?|"
                r"(?:别|不要|不用)(?:发送|转发)(?:任何)?(?:内容|消息|文件))"
            )
            following = re.sub(r"[，,。；;\s]*(?:" + stop + r")[。\s]*$", "", following)
            if re.fullmatch(lead, leading) and re.fullmatch(singular, following):
                return True
        # These complete forms bind the singular reference to the two named
        # candidates. Full matching leaves no room for a third task or target.
        if re.fullmatch(
            cls._REFERENTIAL_CHOICE_CONTRACT.format(target="APP"),
            normalized,
        ):
            return True
        choice = cls._EXPLICIT_ONE_OF_PATTERN.pattern + r"|选一个|挑一个"
        match = re.search(
            r"APP\s*(?:或者|或是|或|和|与|跟|、)\s*APP\s*"
            r"(?:中|里面|里)?(?:的)?\s*[，,]?\s*(?:请|帮我|给我)?\s*"
            r"(?:打开|启动|进入)?\s*(?:" + choice + r")"
            r"(?:应用|App)?(?=$|[，,。！？!?；;\s]|就|即可|便|打开|启动|进入|切)"
            r"\s*(?:打开|启动|进入)?",
            normalized,
        )
        if not match:
            return False
        remaining = re.sub(
            r"^\s*[，,]?\s*(?:打开|进入|启动)\s*(?:任意|其中)?一个(?:即可|就行|就好)?",
            "", normalized[match.end():],
        )
        return not cls._FORWARD_TARGET_LAUNCH_CUE_PATTERN.search(remaining)

    @classmethod
    def _parse_multiple_app_launch(cls, text: str) -> FastPathParserStatus | None:
        """Bind every target in a complete sequence; this never authorizes Fast."""
        if "\x00" in text:
            return None
        if cls._is_explicit_app_choice(text, cls._find_allowed_app_mentions(text)):
            return None
        for boundary in re.finditer(r"[。；;]", text):
            background, current = text[:boundary.start()], text[boundary.end():].strip()
            actions = tuple(cls._HIGH_IMPACT_ACTION_PATTERN.finditer(background))
            if (
                re.match(r"(?:现在|当前)", current) and actions
                and re.match(r"(?:我|我们|他|她)曾(?:经)?(?:用|通过)邮件(?:把|将)", background)
                and not cls._LAUNCH_CUE_PATTERN.search(background)
                and all(cls._action_is_read_content(background, m.start(), m.end()) for m in actions)
            ):
                # Only routing analysis is scoped: the original complete text
                # is still passed to both the model and Slow execution.
                return cls._parse_multiple_app_launch(current)
        declared = re.fullmatch(
            r"能用(?P<first>[^，,；;。]+)也能用(?P<second>[^，,；;。]+)[，,；;]"
            r"(?:此次|这次)(?:分别|依次)(?:启动|打开)(?:这两个|这两款)[。]?",
            text,
        ) or re.fullmatch(
            r"(?:开|打开)(?P<first>[^，,；;。]+?)和(?:开|打开)(?P<second>[^，,；;。]+?)"
            r"都执行[，,]两个都要启动[。]?", text,
        )
        if declared:
            return cls._parse_multiple_app_launch("分别启动" + declared["first"] + "和" + declared["second"])
        declared_all = re.fullmatch(
            r"(?P<targets>[^，,。；;]+)是两(?:个|款)明确候选[，,]"
            r"现在两款全部启动[，,]不要二选一[。]?", text,
        )
        if declared_all:
            return cls._parse_multiple_app_launch("依次启动" + declared_all["targets"])
        separator = r"(?:[，,。；;\n]+|以后再|后再|然后|接着|随后|最后|并且|再|并)"
        bulk = r"(?:(?:分别|依次|逐个|挨个|各自|各|全部|全都|都|一起|两个|两款|这两个|这两款)\s*)*"
        prefix = r"(?:(?:请|麻烦|帮我|替我|现在|这次|先|再|只|然后|接着|随后|最后)\s*)*" + bulk
        forward = r"(?:打开|启动|进入|唤起|切到|切换到|切换至|切回|回到|调出|调出来|叫出|显示一下)"
        backward = (
            r"(?:打开|启动|进入|唤起|开起来|调出|调出来|"
            r"(?:切|切换|移|放|调)(?:到|至)前台|放到最前面|置于最上层|"
            + cls._STATE_LAUNCH_VERB + r")"
        )
        stop = (
            r"(?:(?:各自|分别|都)?(?:的|到|停在)?(?:首页|主页|前台)(?:即停|就停|就好)?|"
            r"(?:就|即|便)(?:停|停止|结束|完成|可)|就行|就好|即可|"
            r"(?:两个|两款)(?:都要|都打开才完成)|一个也别漏|本身|一下|一次|吧)"
        )
        # Non-greedy full target slots retain unknown peers and alias suffixes.
        # Only command separators or bounded stop tails may follow a slot.
        targets = r"[^。！？!?；;\n]+?"
        command = re.compile(
            prefix + r"(?:"
            + forward + r"\s*(?P<forward>" + targets + r")|"
            r"(?:把|将|让)?\s*(?P<backward>" + targets + r")\s*[，,]?\s*"
            + bulk + backward + r")\s*(?:" + stop + r"\s*)*"
            r"(?=$|[。！？!?；;\n]|" + separator + r")"
        )
        remaining = text.strip().rstrip("。！？!?；;")
        initial_mentions = cls._find_allowed_app_mentions(remaining)
        if not initial_mentions:
            return None
        if not (
            re.match(prefix + r"(?:" + forward + r"|把|将|让)", remaining)
            or initial_mentions[0][0] == 0
            or re.fullmatch(r"[^，,。；;\n]{1,24}(?:和|与|及|、|跟)\s*", remaining[:initial_mentions[0][0]])
        ):
            return None
        bound_targets = []
        unknown_target = False
        ambiguous_choice = False
        while remaining:
            # A page/content action stays with the existing read parser. It is
            # not another app, even when followed by a real later app launch.
            if re.match(
                forward + r"\s*(?:" + cls._SUPPORTED_PAGE_TARGET + r")"
                r"(?=$|[，,。；;\n]|后|再|然后)", remaining,
            ):
                return None
            match = command.match(remaining)
            # A later bare name inherits the preceding launch, not an arbitrary
            # query/action. Keep the full name, including unknown suffixes.
            if match is None and bound_targets and (
                any(start == 0 for start, _, _ in cls._find_allowed_app_mentions(remaining))
                or re.match(r"[A-Z][A-Za-z0-9_-]+(?:$|[。；;])", remaining)
            ):
                match = re.match(
                    r"(?P<forward>" + targets + r")\s*(?:" + stop + r"\s*)*"
                    r"(?=$|" + separator + r")", remaining,
                )
            if match is None:
                if unknown_target and len(bound_targets) >= 2:
                    return FastPathParserStatus.UNKNOWN_TARGET
                return None
            target_text = match.groupdict().get("forward") or match["backward"]
            target_mentions = cls._find_allowed_app_mentions(target_text)
            if (
                target_mentions and target_mentions[0][0] == 0
                and re.match(r"(?:的|里|中|内|上)", target_text[target_mentions[0][1]:])
                and re.search(cls._READ_ACTION + r"|进入|打开|收藏", target_text[target_mentions[0][1]:])
            ) or re.match(r"关于.+的(?:回答|文章|内容)", target_text):
                return None
            if match.groupdict().get("backward") and re.match(
                r"我|你|他|她|昨天|刚才|今天|等|先|不|明天|现在|在|用|从|到|去|帮|请|麻烦|为了|关于",
                target_text,
            ):
                return None
            # Protect canonical names such as 恋与深空 before splitting peers.
            tokenized = target_text
            names = {}
            for index, (start, end, _) in reversed(list(enumerate(target_mentions))):
                token = f"\x00{index}\x00"
                names[token] = target_text[start:end]
                tokenized = tokenized[:start] + token + tokenized[end:]
            ambiguous_choice |= bool(re.search(r"还是|或者|或是", tokenized))
            targets_here = re.split(r"\s*(?:以及|还是|或者|或是|和|与|及|、|跟|，|,)\s*", tokenized)
            targets_here = [
                re.sub(r"\x00[0-9]+\x00", lambda m: names[m[0]], part)
                for part in targets_here
            ]
            if bound_targets and any(re.fullmatch(
                cls._SUPPORTED_PAGE_TARGET + r"|(?:里面的|我的)?(?:内容|酒店详情|文档|文件|歌曲)",
                target.strip(),
            ) for target in targets_here):
                return FastPathParserStatus.UNKNOWN_TARGET if unknown_target else None
            bound_targets.extend(targets_here)
            for target in targets_here:
                target = re.sub(
                    r"(?:这个应用|这款应用|应用|软件|程序|客户端|App)$", "",
                    target.strip(), flags=re.IGNORECASE,
                ).strip()
                mentions = cls._find_allowed_app_mentions(target)
                unknown_target |= len(mentions) != 1 or mentions[0][:2] != (0, len(target))
            remaining = remaining[match.end():].strip()
            if not remaining:
                break
            delimiter = re.match(separator + r"\s*", remaining)
            if delimiter is None:
                return None
            remaining = remaining[delimiter.end():].strip()
            boundary = (
                stop + r"|只启动|不查看路线|(?:不要|不用|无需)(?:再|继续)?"
                r"(?:搜索|查询|查看内容|发送结果|删除文件|下单|支付|操作)"
            )
            if re.fullmatch(r"(?:" + boundary + r")(?:[，,]\s*(?:" + boundary + r"))*", remaining):
                remaining = ""
        if len(bound_targets) < 2:
            if unknown_target and bound_targets:
                target = bound_targets[0].strip()
                mentions = cls._find_allowed_app_mentions(target)
                if mentions and mentions[0][0] == 0 and re.fullmatch(
                    r"(?:助手|轻聊版|极速版|专业版|测试版|浏览器|工具)(?:应用|软件|程序|App)?",
                    target[mentions[0][1]:], re.IGNORECASE,
                ):
                    return FastPathParserStatus.UNKNOWN_TARGET
            return None
        if unknown_target:
            return FastPathParserStatus.UNKNOWN_TARGET
        if ambiguous_choice:
            return FastPathParserStatus.AMBIGUOUS_TARGET
        return FastPathParserStatus.KNOWN_COMPLEX_TASK

    @classmethod
    def _has_multiple_app_launch_operations(
        cls,
        text: str,
        mentions: tuple[tuple[int, int, str], ...],
    ) -> bool:
        """Identify separate launches for scope analysis, not a Reject policy."""
        if len(mentions) < 2:
            return False
        # Explicit selection of one app is not a request to launch every app.
        if cls._is_explicit_app_choice(text, mentions):
            return False
        tail = text[mentions[-1][1]:]
        if re.match(
            r"\s*(?:分别|逐个|挨个|都|全部|全都|一起)\s*"
            rf"(?:给我)?(?:打开|启动|调出来|叫出来|唤起|切到前台|回到前台|{cls._STATE_LAUNCH_VERB})",
            tail,
        ):
            return True
        if re.search(r"(?:分别|逐个|挨个)\s*(?:把|将)?\s*$", text[:mentions[0][0]]) and re.match(
            rf"\s*(?:打开|启动|调出来|叫出来|唤起|切到前台|回到前台|{cls._STATE_LAUNCH_VERB})", tail
        ):
            return True
        contract_count = sum(
            bool(
                cls._find_launch_contracts(text, mention)
                or cls._BOUNDED_ATOMIC_LAUNCH_PATTERN.search(
                    cls._normalize_contract_text(text, mention)
                )
            )
            for mention in mentions
        )
        if contract_count > 1:
            return True

        first_start = mentions[0][0]
        has_leading_launch = any(
            cue.end() <= first_start
            for cue in cls._FORWARD_TARGET_LAUNCH_CUE_PATTERN.finditer(text)
        )
        has_coordinated_targets = any(
            re.fullmatch(r"\s*(?:和|与|及|、)\s*", text[left[1] : right[0]])
            for left, right in zip(mentions, mentions[1:])
        )
        return contract_count == 1 and has_leading_launch and has_coordinated_targets

    @classmethod
    def _find_allowed_app_mentions(cls, text: str) -> tuple[tuple[int, int, str], ...]:
        candidates: list[tuple[int, int, str]] = []
        for alias, canonical in cls._ALIASES:
            for match in re.finditer(re.escape(alias), text, re.IGNORECASE):
                candidate = (match.start(), match.end(), canonical)
                if candidate not in candidates:
                    candidates.append(candidate)

        selected: list[tuple[int, int, str]] = []
        for candidate in sorted(
            candidates, key=lambda item: (item[0], -(item[1] - item[0]), item[2])
        ):
            if any(candidate[0] < end and candidate[1] > start for start, end, _ in selected):
                continue
            selected.append(candidate)
        return tuple(sorted(selected))

    @classmethod
    def _is_negated_content_navigation(cls, text, cue, mentions) -> bool:
        """Only a named content noun in an existing app can bind this veto."""
        if not any(end <= cue.start() for _, end, _ in mentions):
            return False
        if not cls._action_is_negated(text, cue.start(), cue.end()):
            return False
        content = (
            r"(?:(?:应用)?里面的|内部的|这些|那份|这份|任何)?"
            r"(?:酒店|歌曲|路线|报道|邮件|文档|对话|书籍|房源|关卡|车次|行情|"
            r"视频|商品|内容|收藏|附件|资料)(?:详情)?"
        )
        after = cls._ACTION_CLAUSE_SEPARATOR.split(text[cue.end():])[0].strip()
        if re.fullmatch(content + r"(?:了)?", after):
            return True
        before = cls._ACTION_CLAUSE_SEPARATOR.split(text[:cue.start()])[-1].strip()
        return not after and bool(re.fullmatch(
            content + r"(?:先|暂时)?(?:不要|不用|不必|别)(?:再|继续)?", before
        ))

    @classmethod
    def _is_read_followup_target(cls, text, cue, target, mentions) -> bool:
        """Bind content/page nouns and singular references to prior app work."""
        if len(mentions) != 1 or mentions[0][1] > cue.start():
            return False
        context = text[mentions[0][1]:cue.start()]
        if not re.search(cls._READ_ACTION + r"|里|中|内|收藏|历史|记录", context):
            return False
        boundary = r"(?=$|\s|" + cls._ACTION_CLAUSE_SEPARATOR.pattern + r"|后|打开|进入|查看|阅读|找到|看|只)"
        page = (
            r"(?:影片|剧集|正片|歌曲|单集|店铺|地点|商户|房型|户型|条目|作品|"
            r"选集|活动规则|媒体|歌手|作者|全部节目)"
            r"(?:营业时间|信息)?(?:详情页?|页面|页|主页)?"
        )
        location = (
            r"(?:我的)?(?:观看历史|浏览历史|收听历史|浏览足迹|观影记录|想读列表|"
            r"稍后再看列表|订阅列表|收藏(?:列表|店铺)?|最近浏览|最近播放|追剧列表)|"
            r"(?:版本|活动)公告|全文|图文筛选|时间排序|用户|"
            r"收藏(?:里|的|里的)[^，,。！？!?；;]{1,20}(?:分组|清单)"
        )
        content = (
            r"(?:公开)?(?:帖子|视频|文档|文章|报道|邮件|歌曲|图文|教程|作品|节目|"
            r"评论|回答|经验分享|内容|聊天记录)"
            r"(?:信息|说明)?(?:详情页?|页面|页)?"
        )
        # Descriptive phrases end in a content role or a counted reference;
        # a free-standing product name such as an unknown notebook app cannot.
        reference = (
            r"(?:(?:最近|近期|最新|对应)(?:更新)?(?:的)?|"
            r"[^，,。！？!?；;]{1,24}的)"
            r"(?:[^，,。！？!?；;]{0,16}" + content
            + r"|一(?:份|封|家|条|单|篇文章)(?:的路线)?(?:详情)?|说明页面|详情页?)|"
            r"(?:最早一班|店铺营业时间|活动规则)(?:详情|页面)|"
            r"(?:那家|那款)[^，,。！？!?；;]{1,20}"
        )
        if re.match(r"(?:" + page + r"|" + location + r"|" + reference + r")" + boundary, target):
            return True
        if re.match(r"[“\"][^”\"]+[”\"]" + boundary, target) and re.search(r"歌单|播放列表|文章|邮件|文档", context):
            return True
        return bool(
            re.match(r"(?:阅读)?" + boundary, target)
            and re.search(r"找到|翻到|搜索", context)
        )

    @classmethod
    def _complete_app_target(cls, text: str) -> bool:
        """A target is a full alias with optional existing app descriptors."""
        target = text.strip()
        mentions = cls._find_allowed_app_mentions(target)
        if len(mentions) != 1 or mentions[0][0] != 0:
            return False
        # Validate the real alias span before applying the existing suffix
        # grammar. Literal user text before the alias cannot collapse with
        # the internal APP marker during descriptor normalization.
        suffix = target[mentions[0][1]:]
        return not suffix or bool(cls._APP_DESCRIPTOR_PATTERN.fullmatch("APP" + suffix))

    @classmethod
    def _is_complete_forward_app_target(cls, text: str, target_start: int) -> bool:
        """Require a whole alias or a syntactically bound action after it."""
        target = re.split(
            cls._ACTION_CLAUSE_SEPARATOR.pattern
            + r"|(?:后|再)(?=" + cls._FORWARD_TARGET_LAUNCH_CUE_PATTERN.pattern + r")",
            text[target_start:], maxsplit=1,
        )[0].strip()
        if cls._complete_app_target(target):
            return True
        mentions = cls._find_allowed_app_mentions(target)
        if not mentions or mentions[0][0] != 0:
            return False
        normalized = cls._normalize_contract_text(target, mentions[0])
        suffix = normalized[3:].strip()
        if cls._suffix_is_safe(suffix):
            return True
        # A verb inside an unknown launch name is not an app-slot boundary.
        # Require an explicit in-app scope before the following predicate;
        # ordinary 在/用 APP queries keep their existing read-context parser.
        scope = re.match(r"^(?:的|里面|里|中|内|上)\s*", suffix)
        if scope is None:
            return False
        suffix = suffix[scope.end():]
        return bool(
            cls._KNOWN_COMPLEX_ACTION_PATTERN.match(suffix)
            or cls._FORWARD_TARGET_LAUNCH_CUE_PATTERN.match(suffix)
            or re.match(
                cls._SUPPORTED_PAGE_TARGET
                + r"(?=$|\s|后|再|" + cls._READ_ACTION + r")", suffix
            )
        )

    @classmethod
    def _has_unbound_launch_target(
        cls,
        text: str,
        mentions: tuple[tuple[int, int, str], ...],
        allow_page_targets: bool = False,
        require_complete_app_targets: bool = False,
    ) -> bool:
        """Reject forward Launch predicates whose target is not a known mention."""
        require_complete_app_targets |= bool(re.search(
            r"(?:" + cls._FORWARD_TARGET_LAUNCH_CUE_PATTERN.pattern + r")\s*"
            + cls._SUPPORTED_PAGE_TARGET + r"(?=$|[\s，,。！？!?；;：:]|然后|再|后)", text
        ))
        for cue in cls._FORWARD_TARGET_LAUNCH_CUE_PATTERN.finditer(text):
            if any(
                (match := cls._BOUNDED_ATOMIC_LAUNCH_PATTERN.match(
                    cls._normalize_contract_text(text, mention), cue.start()
                )) is not None
                and re.match(r"$|[\s，,。！？!?；;]", cls._normalize_contract_text(text, mention)[match.end():])
                for mention in mentions if mention[0] >= cue.start()
            ):
                continue
            target = re.sub(r"^\s*(?:一下)?\s*", "", text[cue.end():])
            app_role = re.match(
                r"(?:(?:这|那)(?:个|款))?(?P<name>[^，,。！？!?；;]{1,32}?)"
                r"(?:应用|软件|程序|客户端|App)(?=$|[\s，,。！？!?；;])",
                target, re.IGNORECASE,
            )
            if app_role:
                name = re.sub(r"(?:这|那)(?:一)?(?:个|款)$", "", app_role["name"].strip()).strip()
                bound = cls._find_allowed_app_mentions(name)
                if len(bound) != 1 or bound[0][:2] != (0, len(name)):
                    return True
                continue

            if cls._is_negated_content_navigation(text, cue, mentions):
                continue
            if any(
                mention_end <= cue.start()
                and re.fullmatch(
                    r"\s*(?:(?:这个应用|那款软件|这个\s*App|那个程序|这个App|App|应用|软件|程序|界面)\s*)*"
                    r"\s*(?:现在|马上)?\s*(?:[，,]\s*)?(?:帮我|给我|替我|为我|帮忙|麻烦)?\s*",
                    text[mention_end : cue.start()],
                    re.IGNORECASE,
                )
                for _mention_start, mention_end, _canonical in mentions
            ):
                continue

            target_start = cue.end()
            optional = re.match(r"\s*(?:一下)?\s*", text[target_start:])
            if optional is not None:
                target_start += optional.end()
            if any(
                mention_start == target_start
                for mention_start, _mention_end, _canonical in mentions
            ):
                if require_complete_app_targets and not cls._is_complete_forward_app_target(text, target_start):
                    return True
                continue

            if allow_page_targets and mentions:
                # A Slow model may resolve in-app articles/items that the
                # narrow Fast parser cannot bind. Do not confuse a named
                # unknown app (AuroraBoard, 云雀笔记) with such referents.
                target = text[target_start:]
                if re.match(cls._SUPPORTED_PAGE_TARGET, target) or re.match(
                    r"我(?:收藏|收到|保存|看过)的|刚|昨天|前天|上次|最新|"
                    r"一(?:个|条|篇|家|套|款|本|首|集|封)|那(?:个|条|篇)|这(?:个|条|篇)|"
                    r"内容|简介|视频介绍|产品详情|查看", target
                ):
                    continue
                if cls._is_read_followup_target(text, cue, target, mentions):
                    continue

            if cls._is_bound_in_app_page_target(
                text, cue, target_start, mentions
            ):
                continue
            return True
        return False

    @classmethod
    def _is_bound_in_app_page_target(
        cls,
        text: str,
        cue: re.Match[str],
        target_start: int,
        mentions: tuple[tuple[int, int, str], ...],
    ) -> bool:
        """Recognize a bounded page action belonging to one preceding known app."""
        # Page ownership cannot legalize an earlier partial app-name match.
        prefix = re.sub(r"(?:然后|接着|随后|以后|后|再)\s*$", "", text[:cue.start()])
        if cls._has_unbound_launch_target(
            prefix, cls._find_allowed_app_mentions(prefix), require_complete_app_targets=True
        ):
            return False
        # A file or duration-qualified video detail belongs to the app after
        # content lookup there. Do not accept a free-standing app-like name.
        if (
            len(mentions) == 1
            and mentions[0][1] <= cue.start()
            and re.search(cls._READ_ACTION, text[mentions[0][1]:cue.start()])
            and re.match(
                r"(?:(?:文件|文档|附件)详情(?:页)?|"
                r"时长[0-9零〇一二两三四五六七八九十百]+(?:秒|分钟|小时)"
                r"(?:以内|以下|以上)的视频详情(?:页)?)"
                r"(?=$|[\s，,。！？!?；;：:]|确认格式(?:$|[\s，,。！？!?；;：:]))",
                text[target_start:],
            )
        ):
            return True
        page_match = re.match(cls._SUPPORTED_PAGE_TARGET, text[target_start:])
        if page_match is None or len(mentions) != 1:
            return False
        target_tail = text[target_start + page_match.end() :]
        if not re.match(r"(?:$|[\s，,。！？!?；;：:]|然后|再|后)", target_tail):
            return False

        mention_start, mention_end, _canonical = mentions[0]
        if mention_end > cue.start():
            return False

        between = text[mention_end : cue.start()]
        if (
            cls._classify_complex_clause(between)
            == FastPathParserStatus.KNOWN_COMPLEX_TASK
        ):
            return True

        if cls._find_launch_contracts(text[: cue.start()], mentions[0]):
            return True

        return bool(
            re.search(r"(?:在|用)\s*$", text[:mention_start])
            and re.fullmatch(
                r"\s*(?:里|中|内|上)?\s*(?:[，,]\s*)?(?:(?:后|然后|再)\s*)?",
                between,
            )
        )

    @classmethod
    def _find_launch_contracts(
        cls, text: str, mention: tuple[int, int, str]
    ) -> list[tuple[str, re.Match[str]]]:
        contract_text = cls._normalize_contract_text(text, mention)
        candidates: list[re.Match[str]] = []
        seen: set[tuple[int, int]] = set()
        for pattern in cls._LAUNCH_CONTRACT_PATTERNS:
            for match in pattern.finditer(contract_text):
                span = match.span()
                if span not in seen:
                    seen.add(span)
                    candidates.append(match)

        selected: list[re.Match[str]] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (-(item.end() - item.start()), item.start(), item.end()),
        ):
            if any(
                candidate.start() < existing.end()
                and candidate.end() > existing.start()
                for existing in selected
            ):
                continue
            selected.append(candidate)
        return [
            (contract_text, match)
            for match in sorted(selected, key=lambda item: (item.start(), item.end()))
        ]

    @classmethod
    def _normalize_contract_text(
        cls, text: str, mention: tuple[int, int, str]
    ) -> str:
        start, end, _canonical = mention
        contract_text = f"{text[:start]}APP{text[end:]}"
        return cls._APP_DESCRIPTOR_PATTERN.sub("APP", contract_text)

    @classmethod
    def _classify_complex_clause(
        cls, text: str
    ) -> FastPathParserStatus | None:
        candidate = text.strip(" \t\r\n，,。！？!?；;：:")
        candidate = re.sub(r"^(?:的|然后|再|帮我|给我|把|在)\s*", "", candidate)
        if cls._INCOMPLETE_COMPLEX_PATTERN.fullmatch(candidate):
            return FastPathParserStatus.UNSUPPORTED_CONTRACT
        if cls._KNOWN_COMPLEX_ACTION_PATTERN.search(candidate):
            return FastPathParserStatus.KNOWN_COMPLEX_TASK
        return None

    @classmethod
    def _prefix_is_safe(cls, prefix: str) -> bool:
        if (
            cls._ALTERNATIVE_PATTERN.search(prefix)
            or cls._CONDITION_OR_NEGATED_LAUNCH.search(prefix)
            or cls._classify_complex_clause(prefix) is not None
        ):
            return False
        clauses = re.split(r"[，,。！？!?；;]", prefix)
        local = clauses[-1]
        return bool(cls._SAFE_LOCAL_PREFIX.fullmatch(local))

    @classmethod
    def _negative_clause_is_safe(cls, clause: str) -> bool:
        """Every content action needs its own local prohibition or coordination."""
        if not cls._SAFE_NEGATIVE_TAIL.fullmatch(clause):
            return False
        return all(
            cls._NEGATED_ACTION_PREFIX.search(clause[:action.start()])
            or cls._NEGATED_CONTENT_CONTINUATION.search(clause[:action.start()])
            for action in cls._KNOWN_COMPLEX_ACTION_PATTERN.finditer(clause)
        )

    @classmethod
    def _suffix_is_safe(cls, suffix: str) -> bool:
        clauses = [part.strip() for part in cls._ACTION_CLAUSE_SEPARATOR.split(suffix)]
        if not clauses:
            return True
        if not (
            cls._SAFE_SUFFIX_PARTICLES.fullmatch(clauses[0])
            or cls._SAFE_PURPOSE_OR_STATE_TAIL.fullmatch(clauses[0])
            or cls._LOCAL_CONTENT_TAIL.fullmatch(clauses[0])
            or cls._negative_clause_is_safe(clauses[0])
        ):
            return False
        return all(
            not clause
            or bool(cls._LOCAL_CONTENT_TAIL.fullmatch(clause))
            or cls._negative_clause_is_safe(clause)
            or bool(cls._SAFE_SELF_TAIL.fullmatch(clause))
            or bool(cls._SAFE_PURPOSE_OR_STATE_TAIL.fullmatch(clause))
            for clause in clauses[1:]
        )


def execute_fast_path_decision(
    decision: FastPathDecision, device_id: str | None = None
) -> tuple[bool, str]:
    """Execute a local Fast Path decision using the existing device factory."""
    if not decision.is_fast_path:
        return False, "Decision is not a Fast Path local action."

    device_factory = get_device_factory()

    if not decision.app_name:
        return False, "Fast launch action missing app name."

    success = device_factory.launch_app(decision.app_name, device_id)
    if success:
        return True, f"Fast Path executed: launch {decision.app_name}"
    return False, f"Fast Path failed to launch app: {decision.app_name}"


def format_fast_path_decision(decision: FastPathDecision) -> str:
    """Return a compact user-facing log line for a Fast Path decision."""
    evidence = (
        f"classifier={decision.classifier_outcome} "
        f"parser={decision.parser_status} guard={decision.guard_outcome}"
    )
    if decision.kind == FastPathDecisionKind.FAST_LAUNCH:
        return f"[Fast Path] fast_launch({decision.app_name}) {evidence}"
    if decision.kind == FastPathDecisionKind.UNSUPPORTED_OR_CLARIFY:
        return f"[Fast Path] unsupported_or_clarify ({decision.reason}) {evidence}"
    return f"[Fast Path] fallback_slow ({decision.reason}) {evidence}"
