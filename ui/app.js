(() => {
  "use strict";

  const LIVE_DEMO_FALLBACK = "https://htinos-multimodal-fashion-recommender.hf.space";

  // Same-origin ("") when served by the FastAPI app itself (mounted at /ui);
  // falls back to the live demo when the file is opened standalone (file://),
  // or can be overridden with ?api=<url>.
  function resolveApiBase() {
    const params = new URLSearchParams(window.location.search);
    if (params.has("api")) return params.get("api").replace(/\/$/, "");
    if (window.location.protocol === "file:") return LIVE_DEMO_FALLBACK;
    return "";
  }

  const API_BASE = resolveApiBase();

  const SUGGESTIONS = [
    "Swim Trunk",
    "Sunglasses",
    "Flip Flop",
    "Denim Jacket",
    "Running Shoes",
    "Wool Coat",
    "Leather Wallet",
    "Graphic T-Shirt",
  ];

  const state = {
    history: [],
  };

  const el = {
    itemInput: document.getElementById("item-input"),
    addBtn: document.getElementById("add-btn"),
    suggestions: document.getElementById("suggestions"),
    history: document.getElementById("history"),
    topK: document.getElementById("top-k"),
    recommendBtn: document.getElementById("recommend-btn"),
    statusLine: document.getElementById("status-line"),
    results: document.getElementById("results"),
    apiInfo: document.getElementById("api-info"),
  };

  function addItem(value) {
    const trimmed = value.trim();
    if (!trimmed) return;
    state.history.push(trimmed);
    renderHistory();
    el.itemInput.value = "";
    el.itemInput.focus();
  }

  function removeItem(index) {
    state.history.splice(index, 1);
    renderHistory();
  }

  function renderSuggestions() {
    el.suggestions.innerHTML = "";
    for (const s of SUGGESTIONS) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip suggestion";
      chip.textContent = `+ ${s}`;
      chip.addEventListener("click", () => addItem(s));
      el.suggestions.appendChild(chip);
    }
  }

  function renderHistory() {
    el.history.innerHTML = "";
    state.history.forEach((item, index) => {
      const chip = document.createElement("span");
      chip.className = "chip history-item";
      chip.textContent = item + " ";

      const remove = document.createElement("button");
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove ${item}`);
      remove.textContent = "×";
      remove.addEventListener("click", () => removeItem(index));

      chip.appendChild(remove);
      el.history.appendChild(chip);
    });
    el.recommendBtn.disabled = state.history.length === 0;
  }

  function setStatus(message, isError) {
    el.statusLine.textContent = message || "";
    el.statusLine.classList.toggle("error", Boolean(isError));
  }

  function renderResults(recommendations) {
    el.results.innerHTML = "";
    if (!recommendations.length) {
      setStatus("No recommendations returned.", false);
      return;
    }
    const maxScore = Math.max(...recommendations.map((r) => r.score), 1e-9);

    for (const rec of recommendations) {
      const card = document.createElement("div");
      card.className = "card";

      const thumb = document.createElement("div");
      thumb.className = "thumb" + (rec.image_url ? "" : " no-image");
      if (rec.image_url) {
        thumb.style.backgroundImage = `url("${rec.image_url}")`;
      }
      const rank = document.createElement("span");
      rank.className = "rank";
      rank.textContent = `#${rec.rank}`;
      thumb.appendChild(rank);

      const body = document.createElement("div");
      body.className = "body";

      const title = document.createElement("div");
      title.className = "title";
      title.textContent = rec.title || "(untitled)";
      title.title = rec.title || "";

      const category = document.createElement("div");
      category.className = "category";
      category.textContent = rec.categories || "";

      const scoreRow = document.createElement("div");
      scoreRow.className = "score-row";
      const bar = document.createElement("div");
      bar.className = "score-bar";
      const fill = document.createElement("span");
      fill.style.width = `${Math.max(4, (rec.score / maxScore) * 100)}%`;
      bar.appendChild(fill);
      const scoreValue = document.createElement("span");
      scoreValue.className = "score-value";
      scoreValue.textContent = rec.score.toFixed(3);
      scoreRow.appendChild(bar);
      scoreRow.appendChild(scoreValue);

      body.appendChild(title);
      body.appendChild(category);
      body.appendChild(scoreRow);

      card.appendChild(thumb);
      card.appendChild(body);
      el.results.appendChild(card);
    }
  }

  async function getRecommendations() {
    el.recommendBtn.disabled = true;
    setStatus("Loading recommendations…", false);
    el.results.innerHTML = "";

    try {
      const response = await fetch(`${API_BASE}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          history: state.history,
          top_k: Number(el.topK.value),
        }),
      });

      if (response.status === 503) {
        setStatus(
          "The model hasn't been trained on this deployment yet (503).",
          true
        );
        return;
      }
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setStatus(`Request failed (${response.status}): ${body.detail || "unknown error"}`, true);
        return;
      }

      const data = await response.json();
      setStatus(`${data.recommendations.length} recommendations for: ${data.history.join(", ")}`, false);
      renderResults(data.recommendations);
    } catch (err) {
      setStatus(`Could not reach the API at "${API_BASE || window.location.origin}": ${err.message}`, true);
    } finally {
      el.recommendBtn.disabled = state.history.length === 0;
    }
  }

  el.addBtn.addEventListener("click", () => addItem(el.itemInput.value));
  el.itemInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addItem(el.itemInput.value);
    }
  });
  el.recommendBtn.addEventListener("click", getRecommendations);

  el.apiInfo.textContent = `API: ${API_BASE || window.location.origin}`;

  renderSuggestions();
  renderHistory();
})();
