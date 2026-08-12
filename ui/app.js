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

  const CATEGORIES = window.CATALOG_SUGGESTIONS || [];
  const ALL_ITEMS = CATEGORIES.flatMap((c) => c.items);

  const state = {
    history: [],
    activeCategory: CATEGORIES.length ? CATEGORIES[0].category : null,
  };

  const el = {
    itemInput: document.getElementById("item-input"),
    addBtn: document.getElementById("add-btn"),
    autocomplete: document.getElementById("autocomplete"),
    categoryTabs: document.getElementById("category-tabs"),
    suggestions: document.getElementById("suggestions"),
    history: document.getElementById("history"),
    emptyHint: document.getElementById("empty-hint"),
    topK: document.getElementById("top-k"),
    clearBtn: document.getElementById("clear-btn"),
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
    hideAutocomplete();
    el.itemInput.focus();
  }

  function removeItem(index) {
    state.history.splice(index, 1);
    renderHistory();
  }

  function renderCategoryTabs() {
    el.categoryTabs.innerHTML = "";
    for (const { category } of CATEGORIES) {
      const tab = document.createElement("button");
      tab.type = "button";
      tab.className = "tab" + (category === state.activeCategory ? " active" : "");
      tab.textContent = category;
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", category === state.activeCategory ? "true" : "false");
      tab.addEventListener("click", () => {
        state.activeCategory = category;
        renderCategoryTabs();
        renderSuggestions();
      });
      el.categoryTabs.appendChild(tab);
    }
  }

  function renderSuggestions() {
    el.suggestions.innerHTML = "";
    const active = CATEGORIES.find((c) => c.category === state.activeCategory);
    if (!active) return;
    for (const item of active.items) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip suggestion";
      chip.textContent = `+ ${item}`;
      chip.addEventListener("click", () => addItem(item));
      el.suggestions.appendChild(chip);
    }
  }

  function hideAutocomplete() {
    el.autocomplete.hidden = true;
    el.autocomplete.innerHTML = "";
  }

  function renderAutocomplete(query) {
    const q = query.trim().toLowerCase();
    if (!q) {
      hideAutocomplete();
      return;
    }
    const matches = ALL_ITEMS.filter((item) => item.toLowerCase().includes(q)).slice(0, 8);
    if (!matches.length) {
      hideAutocomplete();
      return;
    }
    el.autocomplete.innerHTML = "";
    for (const item of matches) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "autocomplete-item";
      row.textContent = item;
      row.addEventListener("mousedown", (e) => {
        // mousedown (not click) so this fires before the input's blur hides the list
        e.preventDefault();
        addItem(item);
      });
      el.autocomplete.appendChild(row);
    }
    el.autocomplete.hidden = false;
  }

  function renderHistory() {
    el.history.innerHTML = "";
    if (state.history.length === 0) {
      el.history.appendChild(el.emptyHint);
    } else {
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
    }
    el.recommendBtn.disabled = state.history.length === 0;
    el.clearBtn.disabled = state.history.length === 0;
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

    recommendations.forEach((rec, i) => {
      const card = document.createElement("div");
      card.className = "card";
      card.style.animationDelay = `${Math.min(i, 10) * 35}ms`;

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
    });
  }

  async function getRecommendations() {
    el.recommendBtn.disabled = true;
    el.recommendBtn.classList.add("loading");
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
      el.recommendBtn.classList.remove("loading");
      el.recommendBtn.disabled = state.history.length === 0;
    }
  }

  el.addBtn.addEventListener("click", () => addItem(el.itemInput.value));
  el.itemInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addItem(el.itemInput.value);
    } else if (e.key === "Escape") {
      hideAutocomplete();
    }
  });
  el.itemInput.addEventListener("input", () => renderAutocomplete(el.itemInput.value));
  el.itemInput.addEventListener("focus", () => renderAutocomplete(el.itemInput.value));
  el.itemInput.addEventListener("blur", () => {
    // Delay so a mousedown on an autocomplete row (which prevents default,
    // see renderAutocomplete) still lands before the list disappears.
    setTimeout(hideAutocomplete, 120);
  });

  el.clearBtn.addEventListener("click", () => {
    state.history = [];
    renderHistory();
  });

  el.recommendBtn.addEventListener("click", getRecommendations);

  el.apiInfo.textContent = `API: ${API_BASE || window.location.origin}`;

  renderCategoryTabs();
  renderSuggestions();
  renderHistory();
})();
