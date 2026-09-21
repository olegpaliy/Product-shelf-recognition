const state = {
  file: null,
  sample: null,
  planogram: "auto",
};

const $ = (id) => document.getElementById(id);

async function loadPlanograms() {
  const res = await fetch("/api/planograms");
  const data = await res.json();
  const select = $("planogram-select");
  select.innerHTML = "";
  for (const item of data.planograms || []) {
    const opt = document.createElement("option");
    opt.value = item.id;
    opt.textContent = item.name || item.id;
    select.appendChild(opt);
  }
  select.value = "auto";
  state.planogram = "auto";
  select.addEventListener("change", () => {
    state.planogram = select.value;
  });
}

async function loadSamples() {
  const res = await fetch("/api/samples");
  const data = await res.json();
  const root = $("samples");
  root.innerHTML = "";
  for (const name of data.samples || []) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "sample";
    card.innerHTML = `<img src="/api/samples/${encodeURIComponent(name)}" alt="${name}" /><span>${name}</span>`;
    card.addEventListener("click", () => {
      state.sample = name;
      state.file = null;
      $("file-input").value = "";
      $("upload-name").textContent = "";
      $("btn-analyze-upload").disabled = true;
      $("btn-analyze-sample").disabled = false;
      root.querySelectorAll(".sample").forEach((el) => el.classList.remove("selected"));
      card.classList.add("selected");
    });
    root.appendChild(card);
  }
}

function showError(msg) {
  $("error").textContent = msg || "";
}

function setBusy(busy) {
  $("progress").classList.toggle("hidden", !busy);
  $("btn-analyze-upload").disabled = busy || !state.file;
  $("btn-analyze-sample").disabled = busy || !state.sample;
}

function planogramQuery() {
  const mode = $("planogram-select").value || "auto";
  const expected = ($("expected-brands").value || "").trim();
  const params = new URLSearchParams();
  params.set("planogram", mode);
  if (expected) params.set("expected", expected);
  // If user typed expected brands, force custom unless they picked a fixed scheme
  if (expected && (mode === "auto" || mode === "none")) {
    params.set("planogram", "custom");
  }
  return params;
}

function renderResult(data) {
  $("screen-upload").classList.add("hidden");
  $("screen-result").classList.remove("hidden");

  $("annotated").src = data.annotated_url + "?t=" + Date.now();
  $("det-count").textContent = data.detection_count;
  const plan = data.planogram || {};
  const enabled = plan.enabled !== false && plan.coarse_compliance_pct != null;

  $("plan-source").textContent = enabled
    ? `${plan.planogram_id || "—"} (${plan.source || data.planogram_mode || "—"})`
    : `не застосовано (${plan.source || plan.reason || "none"})`;

  if (enabled) {
    $("compliance").textContent = `${plan.coarse_compliance_pct ?? "—"}% (presence ${plan.presence_pct ?? "—"}%, row ${plan.row_match_pct ?? "—"}%)`;
    $("missing").textContent = (plan.missing || []).join(", ") || "—";
    $("competitor").textContent = (plan.competitor || []).join(", ") || "—";
  } else {
    $("compliance").textContent = "— (немає схеми для цього фото)";
    $("missing").textContent = "—";
    $("competitor").textContent = "—";
  }

  const qa = data.qa || {};
  const badge = $("qa-badge");
  if (qa.valid) {
    badge.className = "badge ok";
    badge.textContent = "Фото придатне";
  } else {
    badge.className = "badge bad";
    badge.textContent = `Фото непридатне: ${qa.reason || "unknown"}`;
  }

  const cacheBadge = $("cache-badge");
  if (data.cached) {
    cacheBadge.className = "badge cache";
    cacheBadge.textContent = "Precomputed (швидке демо)";
  } else {
    cacheBadge.className = "badge hidden";
  }

  const shareList = $("share-list");
  shareList.innerHTML = "";
  const share = data.brand_share || {};
  const counts = data.brand_counts || {};
  Object.keys(share).forEach((brand) => {
    const li = document.createElement("li");
    li.textContent = `${brand}: ${counts[brand]} (${share[brand]}%)`;
    shareList.appendChild(li);
  });

  const shelfList = $("shelf-list");
  shelfList.innerHTML = "";
  (data.shelf_summary || []).forEach((row) => {
    const li = document.createElement("li");
    const brands = Object.entries(row.brands || [])
      .map(([b, c]) => `${b}×${c}`)
      .join(", ");
    li.textContent = `Ряд ${row.shelf}: ${brands || "—"}`;
    shelfList.appendChild(li);
  });

  $("dl-excel").href = data.excel_url;
  $("dl-json").href = data.json_url;
}

async function analyze() {
  showError("");
  setBusy(true);
  try {
    const pq = planogramQuery();
    let res;
    if (state.file) {
      const fd = new FormData();
      fd.append("file", state.file);
      res = await fetch(`/api/analyze?${pq.toString()}`, { method: "POST", body: fd });
    } else if (state.sample) {
      pq.set("sample", state.sample);
      res = await fetch(`/api/analyze?${pq.toString()}`, { method: "POST" });
    } else {
      throw new Error("Оберіть фото або sample");
    }
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || res.statusText);
    }
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    setBusy(false);
  }
}

$("file-input").addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  state.file = file || null;
  state.sample = null;
  document.querySelectorAll(".sample").forEach((el) => el.classList.remove("selected"));
  $("upload-name").textContent = file ? file.name : "";
  $("btn-analyze-upload").disabled = !file;
  $("btn-analyze-sample").disabled = true;
});

$("btn-analyze-upload").addEventListener("click", analyze);
$("btn-analyze-sample").addEventListener("click", analyze);
$("btn-back").addEventListener("click", () => {
  $("screen-result").classList.add("hidden");
  $("screen-upload").classList.remove("hidden");
});

Promise.all([loadSamples(), loadPlanograms()]).catch((err) =>
  showError(err.message || String(err))
);
