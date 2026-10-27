/*
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 DualRoute authors.
 * DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
 */
const elements = {
  healthBadge: document.getElementById("healthBadge"),
  statusBadge: document.getElementById("statusBadge"),
  executeForm: document.getElementById("executeForm"),
  taskInput: document.getElementById("taskInput"),
  submitButton: document.getElementById("submitButton"),
  currentTask: document.getElementById("currentTask"),
  errorBanner: document.getElementById("errorBanner"),
  statusText: document.getElementById("statusText"),
  startedAt: document.getElementById("startedAt"),
  finishedAt: document.getElementById("finishedAt"),
  returnCode: document.getElementById("returnCode"),
  logsPanel: document.getElementById("logsPanel"),
  resultPanel: document.getElementById("resultPanel"),
};

const runtime = {
  pollHandle: null,
  serviceOnline: false,
  lastStatus: null,
  lastLogs: "",
};

function setBadge(element, text, tone) {
  element.textContent = text;
  element.className = `badge badge-${tone}`;
}

function showError(message) {
  elements.errorBanner.hidden = !message;
  elements.errorBanner.textContent = message || "";
}

function formatValue(value, fallback = "-") {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }
  return String(value);
}

function shouldLockInput(status) {
  return !runtime.serviceOnline || status === "starting" || status === "running";
}

function updateDisabledState() {
  const status = runtime.lastStatus?.status || "idle";
  const locked = shouldLockInput(status);
  elements.taskInput.disabled = locked;
  elements.submitButton.disabled = locked;
  elements.submitButton.textContent = locked ? "执行中..." : "开始执行";
}

function updateHealth(online) {
  runtime.serviceOnline = online;
  setBadge(elements.healthBadge, online ? "在线" : "离线", online ? "success" : "danger");
  updateDisabledState();
}

function updateStatusPanel(status) {
  runtime.lastStatus = status;
  const toneMap = {
    idle: "neutral",
    starting: "warning",
    running: "warning",
    succeeded: "success",
    failed: "danger",
  };

  setBadge(
    elements.statusBadge,
    status.status || "unknown",
    toneMap[status.status] || "neutral",
  );
  elements.currentTask.textContent = status.current_text
    ? `当前任务：${status.current_text}`
    : "当前无任务";
  elements.statusText.textContent = formatValue(status.current_text, "无");
  elements.startedAt.textContent = formatValue(status.started_at);
  elements.finishedAt.textContent = formatValue(status.finished_at);
  elements.returnCode.textContent = formatValue(status.return_code);
  elements.resultPanel.textContent = status.result || "等待任务结果...";
  updateDisabledState();
}

function updateLogsPanel(logs) {
  const nearBottom =
    elements.logsPanel.scrollTop + elements.logsPanel.clientHeight + 32 >=
    elements.logsPanel.scrollHeight;
  elements.logsPanel.textContent = logs || "等待任务日志...";
  if (nearBottom) {
    elements.logsPanel.scrollTop = elements.logsPanel.scrollHeight;
  }
  runtime.lastLogs = logs;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const isJson = response.headers.get("content-type")?.includes("application/json");
  const data = isJson ? await response.json() : null;
  if (!response.ok) {
    const message = data?.message || data?.error || `${response.status} ${response.statusText}`;
    const error = new Error(message);
    error.payload = data;
    throw error;
  }
  return data;
}

async function refreshAll() {
  try {
    const [health, status, logs] = await Promise.all([
      requestJson("/health"),
      requestJson("/status"),
      requestJson("/logs"),
    ]);
    updateHealth(health.status === "ok");
    updateStatusPanel(status);
    updateLogsPanel(logs.logs || "");

    if (status.status === "failed") {
      showError("任务执行失败，请查看日志。");
      return;
    }
    if (status.status === "idle" || status.status === "succeeded") {
      showError("");
    }
  } catch (error) {
    updateHealth(false);
    showError(`服务不可用：${error.message}`);
  }
}

async function submitTask(event) {
  event.preventDefault();
  const text = elements.taskInput.value.trim();
  if (!text) {
    showError("请输入任务内容。");
    return;
  }

  showError("");
  try {
    await requestJson("/execute", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
    elements.taskInput.value = text;
    await refreshAll();
  } catch (error) {
    if (error.payload?.status === "busy") {
      showError("当前已有任务执行中，请等待完成后再提交。");
    } else {
      showError(`提交失败：${error.message}`);
    }
    await refreshAll();
  }
}

function startPolling() {
  if (runtime.pollHandle !== null) {
    clearInterval(runtime.pollHandle);
  }
  runtime.pollHandle = window.setInterval(refreshAll, 1000);
}

elements.executeForm.addEventListener("submit", submitTask);
window.addEventListener("load", async () => {
  await refreshAll();
  startPolling();
});
