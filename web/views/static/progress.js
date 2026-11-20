// Progress page: asks the server for the job's status every 2 seconds and shows the current stage and output.
(function () {
  const root = document.getElementById("progress");
  if (!root) return;
  const url = root.dataset.statusUrl;
  const message = document.getElementById("state-message");
  const log = document.getElementById("log");
  const link = document.getElementById("report-link");
  const rerun = document.getElementById("rerun");
  const texts = {
    running: "Running. This page updates by itself.",
    done: "Finished.",
    failed: "The analysis stopped with an error.",
    interrupted: "Interrupted: the web app was stopped while this analysis ran. Start it again from New analysis.",
    none: "This lecture's analysis has not been started from here.",
  };

  function render(status) {
    const stage = status.stage || 0;
    document.querySelectorAll("#stages li").forEach(function (item) {
      const number = Number(item.dataset.stage);
      item.classList.remove("done", "current", "failed");
      if (status.state === "done" || number < stage) item.classList.add("done");
      else if (number === stage) item.classList.add(status.state === "failed" || status.state === "interrupted" ? "failed" : "current");
    });
    let text = texts[status.state] || status.state;
    if (status.state === "running" && status.stage_name) text = "Stage " + stage + ": " + status.stage_name + "…";
    if (status.error) text += " " + status.error;
    if (status.llm_result) text += " " + status.llm_result;
    message.textContent = text;
    if (status.log) {
      const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
      log.textContent = status.log.join("\n");
      if (atBottom) log.scrollTop = log.scrollHeight;
    }
    link.hidden = !(status.has_report && status.state === "done");
    if (rerun) rerun.hidden = !(status.state === "failed" || status.state === "interrupted");
    return status.state === "running" || status.state === "queued";
  }

  function poll() {
    fetch(url, { cache: "no-store" })
      .then(function (response) { return response.json(); })
      .then(function (status) { if (render(status)) setTimeout(poll, 2000); })
      .catch(function () { setTimeout(poll, 5000); });
  }
  poll();
})();
