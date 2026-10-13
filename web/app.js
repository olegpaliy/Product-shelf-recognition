const state = {
  file: null,
  sample: null,
  hasResult: false,
  result: null,
  selectedIndex: null,
  mode: "ok", // ok | change | new | unknown | not_product
  naturalW: 0,
  naturalH: 0,
};

const NAV_TITLES = {
  share: "Нумерична дистрибуція",
  details: "Нумерична дистрибуція (деталі)",
  planogram: "Планограма ХО",
  missing: "SKU не в планограмі",
  competitor: "Конкурент в ХО",
  visits: "Розрахунок % візитів з фото і валідних фото",
};

const RESULT_NAV = new Set(["share", "details", "missing", "competitor"]);

const $ = (id) => document.getElementById(id);

function setNavEnabled(hasResult) {
  state.hasResult = hasResult;
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.disabled = !hasResult;
  });
  document.querySelector(".app-shell")?.classList.toggle("nav-hidden", !hasResult);
}

function setActiveNav(key) {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.nav === key);
  });
  const title = NAV_TITLES[key] || NAV_TITLES.planogram;
  const h1 = document.querySelector(".topbar h1");
  if (h1) h1.textContent = title;
}

function highlightPanel(targetId) {
  document.querySelectorAll(".metric-panel").forEach((el) => el.classList.remove("flash"));
  const panel = document.getElementById(targetId);
  if (!panel) return;
  panel.classList.add("flash");
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function onNavClick(btn) {
  const key = btn.dataset.nav;
  const target = btn.dataset.target;
  setActiveNav(key);

  if (!state.hasResult) {
    if (RESULT_NAV.has(key)) {
      showError("Спочатку проаналізуйте фото");
    }
    $("screen-result").classList.add("hidden");
    $("screen-upload").classList.remove("hidden");
    $("btn-back").classList.add("hidden");
    return;
  }

  showError("");
  $("screen-upload").classList.add("hidden");
  $("screen-result").classList.remove("hidden");
  $("btn-back").classList.remove("hidden");
  if (target) highlightPanel(target);
}

function bindNav() {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => onNavClick(btn));
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
      $("upload-name").textContent = name;
      root.querySelectorAll(".sample").forEach((el) => el.classList.remove("selected"));
      card.classList.add("selected");
      updateAnalyzeEnabled();
    });
    root.appendChild(card);
  }
}

function showError(msg) {
  $("error").textContent = msg || "";
}

function canAnalyze() {
  return !!(state.file || state.sample);
}

function updateAnalyzeEnabled(busy = false) {
  $("btn-analyze").disabled = busy || !canAnalyze();
}

function setBusy(busy) {
  $("progress").classList.toggle("hidden", !busy);
  updateAnalyzeEnabled(busy);
}

function analyzeQuery() {
  const params = new URLSearchParams();
  params.set("planogram", "auto");
  return params;
}

function brandClass(brand, expectedSet, competitorSet) {
  const b = String(brand || "").toLowerCase();
  if (competitorSet.has(b)) return "competitor";
  if (expectedSet.size && !expectedSet.has(b) && b !== "unknown" && b !== "other") {
    return "competitor";
  }
  if (b === "unknown" || b === "other" || !b) return "unknown";
  return "own";
}

function shelfNum(key) {
  if (typeof key === "number" && Number.isFinite(key)) return key;
  const m = String(key).match(/(\d+)\s*$/);
  return m ? Number(m[1]) : NaN;
}

function flattenExpected(expected) {
  if (!expected) return [];
  if (Array.isArray(expected)) return expected;
  if (typeof expected === "object") {
    return Object.values(expected).flat();
  }
  return [];
}

function rowsFromPayload(data) {
  const summary = data.shelf_summary || [];
  if (summary.length) {
    return summary.map((row) => ({
      shelf: Number(row.shelf),
      brands: row.brands || {},
    }));
  }

  const plan = data.planogram || {};
  const byRow = plan.detected_by_row;
  if (byRow && typeof byRow === "object" && !Array.isArray(byRow)) {
    return Object.keys(byRow)
      .map((k) => ({ key: k, shelf: shelfNum(k) }))
      .filter((x) => Number.isFinite(x.shelf))
      .sort((a, b) => a.shelf - b.shelf)
      .map(({ key, shelf }) => {
        const raw = byRow[key] ?? [];
        const brands = {};
        if (Array.isArray(raw)) {
          raw.forEach((b) => {
            const name = typeof b === "string" ? b : b?.brand || b?.name;
            if (!name) return;
            brands[name] = (brands[name] || 0) + 1;
          });
        } else if (raw && typeof raw === "object") {
          Object.assign(brands, raw);
        }
        return { shelf, brands };
      });
  }
  return [];
}

function renderCooler(data) {
  const root = $("cooler");
  root.innerHTML = "";
  const plan = data.planogram || {};
  const expected = new Set(
    flattenExpected(plan.expected_brands || plan.expected).map((b) => String(b).toLowerCase())
  );
  const competitor = new Set((plan.competitor || []).map((b) => String(b).toLowerCase()));
  const missing = (plan.missing || []).map((b) => String(b));
  const rows = rowsFromPayload(data);

  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "cooler-empty";
    empty.textContent = "Немає рядів для схеми ХО";
    root.appendChild(empty);
    return;
  }

  const maxShelf = Math.max(...rows.map((r) => Number(r.shelf) || 0), rows.length);
  const byShelf = new Map(rows.map((r) => [Number(r.shelf), r]));

  for (let shelf = 1; shelf <= maxShelf; shelf += 1) {
    const row = byShelf.get(shelf) || { shelf, brands: {} };
    const shelfEl = document.createElement("div");
    shelfEl.className = "cooler-shelf";

    const label = document.createElement("div");
    label.className = "cooler-shelf-label";
    label.textContent = `Полиця ${shelf}`;
    shelfEl.appendChild(label);

    const chips = document.createElement("div");
    chips.className = "cooler-chips";

    const entries = Object.entries(row.brands || {});
    if (!entries.length) {
      const chip = document.createElement("span");
      chip.className = "chip unknown";
      chip.textContent = "порожньо";
      chips.appendChild(chip);
    } else {
      entries
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
        .forEach(([brand, count]) => {
          const chip = document.createElement("span");
          chip.className = `chip ${brandClass(brand, expected, competitor)}`;
          chip.textContent = count > 1 ? `${brand} ×${count}` : brand;
          chips.appendChild(chip);
        });
    }

    // Show missing expected brands only on first shelf as dashed chips if nowhere detected
    if (shelf === 1 && missing.length) {
      missing.forEach((brand) => {
        const chip = document.createElement("span");
        chip.className = "chip missing";
        chip.textContent = `${brand} (немає)`;
        chips.appendChild(chip);
      });
    }

    shelfEl.appendChild(chips);
    root.appendChild(shelfEl);
  }
}

function setCorrectStatus(msg) {
  $("correct-status").textContent = msg || "";
}

function clearSelection() {
  state.selectedIndex = null;
  state.mode = "ok";
  $("correct-body").classList.add("hidden");
  $("correct-hint").classList.remove("hidden");
  $("btn-correct-save").disabled = true;
  setCorrectStatus("");
  document.querySelectorAll(".bbox-hit").forEach((el) => el.classList.remove("selected"));
}

async function fillBrandSelect(brands) {
  const select = $("correct-brand-select");
  const current = select.value;
  select.innerHTML = "";
  const list = [...new Set(brands || [])].sort();
  if (!list.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "— немає брендів —";
    select.appendChild(opt);
    return;
  }
  list.forEach((brand) => {
    const opt = document.createElement("option");
    opt.value = brand;
    opt.textContent = brand;
    select.appendChild(opt);
  });
  if (current && list.includes(current)) select.value = current;
}

async function refreshCatalogSelect() {
  try {
    const res = await fetch("/api/catalog");
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const brands = (data.brands || []).map((b) => b.brand);
    const fromResult = state.result?.catalog_brands || [];
    await fillBrandSelect([...brands, ...fromResult]);
  } catch (err) {
    await fillBrandSelect(state.result?.catalog_brands || []);
    setCorrectStatus(err.message || String(err));
  }
}

function selectedDetection() {
  if (!state.result || state.selectedIndex == null) return null;
  return state.result.detections?.[state.selectedIndex] || null;
}

function updateCorrectPanel() {
  const det = selectedDetection();
  if (!det) {
    clearSelection();
    return;
  }
  $("correct-hint").classList.add("hidden");
  $("correct-body").classList.remove("hidden");
  const brand = det.brand || "unknown";
  const conf = det.brand_confidence != null ? Number(det.brand_confidence).toFixed(3) : "—";
  $("correct-current").textContent = `${brand} (${conf})`;

  const alts = $("correct-alts");
  alts.innerHTML = "";
  (det.brand_alternatives || []).forEach((alt) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "correct-alt";
    btn.textContent = `${alt.brand} ${Number(alt.score).toFixed(3)}`;
    btn.addEventListener("click", () => {
      state.mode = "change";
      $("correct-brand-select").value = alt.brand;
      $("correct-brand-new").value = "";
      document.querySelectorAll(".correct-alt").forEach((el) => el.classList.remove("active"));
      btn.classList.add("active");
      syncSaveEnabled();
    });
    alts.appendChild(btn);
  });

  if (brand && brand !== "unknown" && brand !== "not_product") {
    const select = $("correct-brand-select");
    if ([...select.options].some((o) => o.value === brand)) select.value = brand;
  }
  syncSaveEnabled();
}

function syncSaveEnabled() {
  const det = selectedDetection();
  if (!det) {
    $("btn-correct-save").disabled = true;
    return;
  }
  if (state.mode === "unknown" || state.mode === "not_product") {
    $("btn-correct-save").disabled = false;
    $("btn-correct-save").textContent = "Застосувати";
    return;
  }
  $("btn-correct-save").textContent = "Зберегти в каталог";
  if (state.mode === "new") {
    $("btn-correct-save").disabled = !$("correct-brand-new").value.trim();
    return;
  }
  // ok / change
  const brand =
    state.mode === "change"
      ? $("correct-brand-select").value
      : det.brand && det.brand !== "unknown"
        ? det.brand
        : $("correct-brand-select").value;
  $("btn-correct-save").disabled = !brand;
}

function layoutBBoxes() {
  const img = $("annotated");
  const layer = $("bbox-layer");
  if (!img || !layer || !state.result) return;
  const dets = state.result.detections || [];
  const nw = state.naturalW || img.naturalWidth;
  const nh = state.naturalH || img.naturalHeight;
  if (!nw || !nh) return;

  if (!img.clientWidth || !img.clientHeight) return;

  layer.innerHTML = "";
  dets.forEach((det, index) => {
    const hit = document.createElement("button");
    hit.type = "button";
    hit.className = "bbox-hit" + (state.selectedIndex === index ? " selected" : "");
    hit.style.left = `${(det.x1 / nw) * 100}%`;
    hit.style.top = `${(det.y1 / nh) * 100}%`;
    hit.style.width = `${((det.x2 - det.x1) / nw) * 100}%`;
    hit.style.height = `${((det.y2 - det.y1) / nh) * 100}%`;
    hit.title = `${det.brand || "?"} ${det.brand_confidence ?? ""}`;
    hit.addEventListener("click", (e) => {
      e.preventDefault();
      state.selectedIndex = index;
      state.mode = "ok";
      $("correct-brand-new").value = "";
      document.querySelectorAll(".bbox-hit").forEach((el) => el.classList.remove("selected"));
      hit.classList.add("selected");
      updateCorrectPanel();
    });
    layer.appendChild(hit);
  });
}

function renderResult(data) {
  state.result = data;
  state.selectedIndex = null;
  state.mode = "ok";

  setNavEnabled(true);
  setActiveNav("planogram");

  $("screen-upload").classList.add("hidden");
  $("screen-result").classList.remove("hidden");
  $("btn-back").classList.remove("hidden");

  const img = $("annotated");
  img.onload = () => {
    state.naturalW = img.naturalWidth;
    state.naturalH = img.naturalHeight;
    layoutBBoxes();
  };
  img.src = data.annotated_url + "?t=" + Date.now();

  $("det-count").textContent = data.detection_count;
  const plan = data.planogram || {};
  const enabled = plan.enabled !== false && plan.coarse_compliance_pct != null;

  $("plan-source").textContent = enabled
    ? `${plan.planogram_id || "—"} (${plan.source || data.planogram_mode || "—"})`
    : `не застосовано (${plan.source || plan.reason || "none"})`;

  if (enabled) {
    $("compliance").textContent = `${plan.coarse_compliance_pct ?? "—"}%`;
    $("compliance").title = `presence ${plan.presence_pct ?? "—"}%, row ${plan.row_match_pct ?? "—"}%`;
    $("missing").textContent = (plan.missing || []).join(", ") || "—";
    $("competitor").textContent = (plan.competitor || []).join(", ") || "—";
  } else {
    $("compliance").textContent = "—";
    $("compliance").title = "немає схеми для цього фото";
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
    const brands = Object.entries(row.brands || {})
      .map(([b, c]) => `${b}×${c}`)
      .join(", ");
    li.textContent = `Ряд ${row.shelf}: ${brands || "—"}`;
    shelfList.appendChild(li);
  });

  renderCooler(data);

  $("dl-excel").href = data.excel_url;
  $("dl-json").href = data.json_url;

  clearSelection();
  refreshCatalogSelect();
  highlightPanel("panel-cooler");
}

async function saveCorrection() {
  const det = selectedDetection();
  if (!det || !state.result) return;
  setCorrectStatus("Збереження…");

  let action = "save";
  let brand = det.brand;

  if (state.mode === "unknown") {
    action = "unknown";
    brand = "unknown";
  } else if (state.mode === "not_product") {
    action = "not_product";
    brand = "not_product";
  } else if (state.mode === "new") {
    brand = $("correct-brand-new").value.trim();
    if (!brand) {
      setCorrectStatus("Вкажіть назву нового бренду");
      return;
    }
    const created = await fetch("/api/catalog/brands", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: brand }),
    });
    if (!created.ok) throw new Error(await created.text());
    const createdData = await created.json();
    brand = createdData.brand;
    action = "save";
  } else if (state.mode === "change") {
    brand = $("correct-brand-select").value;
    action = "save";
  } else {
    // ok — save current brand into catalog
    brand =
      det.brand && det.brand !== "unknown" && det.brand !== "not_product"
        ? det.brand
        : $("correct-brand-select").value;
    action = "save";
  }

  const res = await fetch("/api/catalog/references", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      result_id: state.result.result_id,
      detection_index: state.selectedIndex,
      brand,
      action,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();

  const idx = state.selectedIndex;
  state.result.detections[idx].brand = data.brand;
  if (data.saved) {
    state.result.detections[idx].brand_confidence = 1;
    setCorrectStatus(`Збережено в catalog/${data.brand}/`);
  } else {
    state.result.detections[idx].brand_confidence = 0;
    setCorrectStatus(`Позначено як ${data.brand}`);
  }
  $("correct-current").textContent = `${data.brand}`;
  await refreshCatalogSelect();
  layoutBBoxes();
}

async function reloadCatalogModel() {
  setCorrectStatus("Оновлення embeddings…");
  const res = await fetch("/api/catalog/reload", { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  await fillBrandSelect((data.brands || []).map((b) => b.brand));
  setCorrectStatus("Бренд-модель оновлено (перезапустіть analyze для повного перерахунку полиці)");
}

async function analyze() {
  showError("");
  setBusy(true);
  try {
    const pq = analyzeQuery();
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
  updateAnalyzeEnabled();
});

$("btn-analyze").addEventListener("click", analyze);
$("btn-back").addEventListener("click", () => {
  $("screen-result").classList.add("hidden");
  $("screen-upload").classList.remove("hidden");
  $("btn-back").classList.add("hidden");
  setNavEnabled(false);
  setActiveNav("planogram");
  state.result = null;
  clearSelection();
  $("bbox-layer").innerHTML = "";
  document.querySelectorAll(".metric-panel").forEach((el) => el.classList.remove("flash"));
});

$("btn-correct-ok").addEventListener("click", () => {
  state.mode = "ok";
  $("correct-brand-new").value = "";
  syncSaveEnabled();
  setCorrectStatus("Підтвердження поточного бренду — збережіть у каталог");
});
$("btn-correct-change").addEventListener("click", () => {
  state.mode = "change";
  $("correct-brand-new").value = "";
  syncSaveEnabled();
  setCorrectStatus("Оберіть бренд зі списку або top-3");
});
$("btn-correct-new").addEventListener("click", () => {
  state.mode = "new";
  syncSaveEnabled();
  $("correct-brand-new").focus();
  setCorrectStatus("Вкажіть slug нового бренду");
});
$("btn-correct-unknown").addEventListener("click", () => {
  state.mode = "unknown";
  syncSaveEnabled();
  setCorrectStatus("Позначити як unknown (без каталогу)");
});
$("btn-correct-notproduct").addEventListener("click", () => {
  state.mode = "not_product";
  syncSaveEnabled();
  setCorrectStatus("Позначити як не товар (без каталогу)");
});
$("correct-brand-select").addEventListener("change", () => {
  state.mode = "change";
  syncSaveEnabled();
});
$("correct-brand-new").addEventListener("input", () => {
  state.mode = "new";
  syncSaveEnabled();
});
$("btn-correct-save").addEventListener("click", () => {
  saveCorrection().catch((err) => setCorrectStatus(err.message || String(err)));
});
$("btn-reload-catalog").addEventListener("click", () => {
  reloadCatalogModel().catch((err) => setCorrectStatus(err.message || String(err)));
});
window.addEventListener("resize", () => layoutBBoxes());

bindNav();
setNavEnabled(false);
setActiveNav("planogram");

loadSamples().catch((err) => showError(err.message || String(err)));
