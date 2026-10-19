"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  catalog: null,
  model: "",
  ratings: [],
  sort: "rating",
  direction: -1,
  simulation: "",
  evaluation: null,
  jobs: [],
  page: 0,
  hasDraws: false,
};
const escapeHTML = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const signed = (n) => `${n > 0 ? "+" : ""}${Number(n).toFixed(1)}`;
const percent = (n) => `${(100 * Number(n)).toFixed(1)}%`;
const roundName = (teams, round) => {
  const remaining = teams / 2 ** (round + 1);
  return (
    {
      1: "Champion",
      2: "Final",
      4: "Final Four",
      8: "Elite Eight",
      16: "Sweet 16",
      32: "Round of 32",
      64: "Round of 64",
    }[remaining] || `${remaining} teams remain`
  );
};

const gameRoundName = (teams, round) => {
  const field = teams / 2 ** round;
  return (
    {
      2: "Final",
      4: "Final Four",
      8: "Elite Eight",
      16: "Sweet 16",
      32: "Round of 32",
      64: "Round of 64",
      128: "Round of 128",
    }[field] || `${field} teams`
  );
};
const jobRows = new Map();
function updateJobButtons() {
  const busy = state.jobs.some((job) =>
    ["queued", "running"].includes(job.status),
  );
  $("fit-form").querySelector('button[type="submit"]').disabled =
    busy || !$("season").value;
  $("tournament-form").querySelector('button[type="submit"]').disabled =
    busy || !state.model || !state.hasDraws;
}

async function api(path, body) {
  const options =
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Workspace-Token": document.querySelector(
              'meta[name="workspace-token"]',
            ).content,
          },
          body: JSON.stringify(body),
        };
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok)
    throw new Error(
      result.error || "The request failed. Refresh the page and try again.",
    );
  return result;
}
function showError(error) {
  $("global-error").textContent = error.message || String(error);
  $("global-error").hidden = false;
}
function clearError() {
  $("global-error").hidden = true;
}
async function action(button, label, operation) {
  clearError();
  const old = button.textContent;
  button.disabled = true;
  button.textContent = label;
  try {
    await operation();
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
    if (
      ["fit-form", "tournament-form"].includes(button.form?.id) &&
      state.jobs.some((j) => ["queued", "running"].includes(j.status))
    )
      button.disabled = true;
    button.textContent = old;
  }
}
function selectOptions(
  id,
  options,
  selected = "",
  empty = "No saved items available",
) {
  const select = $(id);
  select.replaceChildren();
  if (!options.length) select.add(new Option(empty, ""));
  for (const option of options) select.add(new Option(option.name, option.id));
  if (options.some((option) => option.id === selected)) select.value = selected;
  select.disabled = !options.length;
}
function download(id, run, file) {
  $(id).href = `/api/download?${new URLSearchParams({ run, file })}`;
  $(id).hidden = !run;
}
function table(id, columns, rows) {
  $(id).innerHTML =
    `<table><thead><tr>${columns.map((c) => `<th scope="col" class="${c.team ? "team" : ""}">${escapeHTML(c.label)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${columns.map((c) => `<td class="${c.team ? "team" : ""}">${escapeHTML(c.format ? c.format(row[c.key], row) : row[c.key])}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}
function navigate() {
  const requested = location.hash.slice(1) || "ratings";
  const view = [
    "ratings",
    "matchups",
    "tournament",
    "backtests",
    "fit",
  ].includes(requested)
    ? requested
    : "ratings";
  for (const section of document.querySelectorAll(".view"))
    section.hidden = section.id !== `view-${view}`;
  for (const link of document.querySelectorAll("nav a")) {
    if (link.dataset.view === view) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  document.title = `${$("view-" + view).querySelector("h1").textContent} · March Madness`;
}
async function catalog(preferred = "") {
  state.catalog = await api("/api/catalog");
  const { fits, backtests, datasets, tournaments, simulations, problems } =
    state.catalog;
  selectOptions(
    "model",
    fits,
    preferred || state.model,
    "No saved season fits — fit a season to begin",
  );
  selectOptions(
    "backtest-run",
    backtests,
    $("backtest-run").value,
    "No saved evaluations",
  );
  selectOptions(
    "dataset",
    datasets,
    $("dataset").value,
    "Add a Parquet game dataset to data/",
  );
  selectOptions(
    "tournament-file",
    tournaments,
    $("tournament-file").value,
    "Add a tournament CSV to data/",
  );
  selectOptions(
    "simulation",
    simulations || [],
    state.simulation,
    "No simulations saved yet",
  );
  $("catalog-problems").textContent = problems.join("\n");
  $("catalog-problems").hidden = !problems.length;
  $("ratings-empty").hidden = !!fits.length;
  $("backtest-empty").hidden = !!backtests.length;
  datasetSeasons();
  await Promise.all([loadRatings(), loadEvaluation()]);
  if ($("simulation").value) await loadSimulation();
}
let ratingsRequest = 0;
async function loadRatings() {
  const request = ++ratingsRequest;
  state.model = $("model").value;
  state.ratings = [];
  state.hasDraws = false;
  state.page = 0;
  renderRatings();
  $("fit-source").textContent = state.model;
  for (const id of ["matchup-context", "tournament-context"])
    $(id).textContent = state.model
      ? `Season fit: ${state.model}. Change the fit on the Team ratings page.`
      : "Fit a season or select a saved fit on the Team ratings page first.";
  for (const id of ["matchup-form", "tournament-form"])
    $(id).querySelector('button[type="submit"]').disabled = !state.model;
  $("diagnostics").replaceChildren();
  $("matchup-result").innerHTML =
    '<p class="muted">Choose two teams to see their win probabilities and expected margin.</p>';
  if (!state.model) {
    selectOptions("first", [], "", "No teams available");
    selectOptions("second", [], "", "No teams available");
    return;
  }
  $("ratings-table").textContent = "Loading ratings…";
  const result = await api(
    `/api/ratings?${new URLSearchParams({ run: state.model })}`,
  );
  if (request !== ratingsRequest) return;
  state.ratings = result.teams;
  state.hasDraws = result.has_draws;
  updateJobButtons();
  download("ratings-download", state.model, "results.csv");
  const teams = [...result.teams]
    .sort((a, b) => a.name.localeCompare(b.name))
    .map((t) => ({ id: String(t.tid), name: t.name }));
  selectOptions("first", teams);
  selectOptions("second", teams, teams[1]?.id);
  $("matchup-form").querySelector("button").disabled = !result.has_draws;
  updateJobButtons();
  const diagnostic = result.diagnostics;
  if (diagnostic) {
    const good =
      diagnostic.max_rhat <= 1.01 &&
      diagnostic.min_bulk_ess >= 400 &&
      !(diagnostic.divergences || diagnostic.max_treedepth_hits);
    $("diagnostics").innerHTML =
      `<span class="${good ? "positive" : "status-warning"}">${good ? "Sampling checks passed" : "Review sampling diagnostics"}</span><span>Max R-hat <strong>${Number(diagnostic.max_rhat).toFixed(3)}</strong></span><span>Min bulk ESS <strong>${Math.round(diagnostic.min_bulk_ess).toLocaleString()}</strong></span>${diagnostic.divergences !== undefined ? `<span>Divergences <strong>${diagnostic.divergences}</strong></span>` : ""}`;
  } else
    $("diagnostics").textContent =
      "No saved diagnostic summary. Review this fit's sampling logs before using forecasts.";
  for (const warning of result.warnings) {
    const p = document.createElement("p");
    p.className = "notice";
    p.textContent = warning;
    $("diagnostics").append(p);
  }
  if (!result.has_draws) {
    const p = document.createElement("p");
    p.className = "notice";
    p.textContent =
      "Ratings only: posterior draws are missing. Refit this season to compare matchups or simulate brackets.";
    $("diagnostics").append(p);
  }
  renderRatings();
}
function renderRatings() {
  const search = $("team-search").value.toLowerCase();
  const ranked = new Map(state.ratings.map((t, i) => [t.tid, i + 1]));
  const filtered = state.ratings
    .filter((t) => t.name.toLowerCase().includes(search))
    .sort(
      (a, b) =>
        state.direction *
        (typeof a[state.sort] === "string"
          ? a[state.sort].localeCompare(b[state.sort])
          : a[state.sort] - b[state.sort]),
    );
  const total = filtered.length;
  state.page = Math.max(0, Math.min(state.page, Math.ceil(total / 25) - 1));
  const rows = filtered.slice(state.page * 25, (state.page + 1) * 25);
  $("ratings-count").textContent = total
    ? `Showing ${state.page * 25 + 1}–${Math.min(total, (state.page + 1) * 25)} of ${total} teams`
    : "";
  $("previous-page").disabled = state.page === 0;
  $("next-page").disabled = (state.page + 1) * 25 >= total;
  if (!rows.length) {
    $("ratings-table").textContent = state.ratings.length
      ? "No teams match this search. Try another name."
      : "";
    return;
  }
  const columns = [
    ["name", "Team"],
    ["rating", "Overall"],
    ["offense", "Offense"],
    ["defense", "Defense"],
    ["home", "Home advantage"],
  ];
  $("ratings-table").innerHTML =
    `<table><thead><tr><th scope="col">Rank</th>${columns.map(([key, label]) => `<th scope="col" class="${key === "name" ? "team" : ""}" ${state.sort === key ? `aria-sort="${state.direction === 1 ? "ascending" : "descending"}"` : ""}><button class="sort" data-sort="${key}">${label}${state.sort === key ? (state.direction === 1 ? " ↑" : " ↓") : ""}</button></th>`).join("")}</tr></thead><tbody>${rows.map((t) => `<tr><td class="rank">${ranked.get(t.tid)}</td><td class="team">${escapeHTML(t.name)}</td><td><span class="rating-value ${t.rating > 0 ? "positive" : ""}">${signed(t.rating)}</span></td><td>${signed(t.offense)}</td><td>${signed(t.defense)}</td><td>${signed(t.home)}</td></tr>`).join("")}</tbody></table>`;
}
function datasetSeasons() {
  const dataset = state.catalog?.datasets.find(
    (d) => d.id === $("dataset").value,
  );
  const seasons = (dataset?.seasons || []).map((s) => ({
    id: `${s.division}|${s.year}`,
    name: `${s.year} · ${s.division === "NCAAM" ? "Men" : "Women"} · ${s.games.toLocaleString()} games in file`,
  }));
  selectOptions("season", seasons, $("season").value, "No seasons found");
  $("dataset-info").textContent =
    "Tournament games are excluded from fitting. Inputs are checked for duplicate game identities.";
  $("fit-form").querySelector('button[type="submit"]').disabled =
    !seasons.length;
}
let evaluationRequest = 0;
async function loadEvaluation() {
  const request = ++evaluationRequest;
  const run = $("backtest-run").value;
  $("backtest-result").hidden = !run;
  download("metrics-download", run, "metrics.csv");
  if (!run) return;
  const result = await api(`/api/backtest?${new URLSearchParams({ run })}`);
  if (request !== evaluationRequest) return;
  state.evaluation = result;
  selectOptions(
    "evaluation-year",
    [...new Set(state.evaluation.metrics.map((m) => String(m.year)))].map(
      (y) => ({ id: y, name: y === "pooled" ? "All seasons" : y }),
    ),
    "pooled",
  );
  renderEvaluation();
}
function renderEvaluation() {
  if (!state.evaluation) return;
  const year = $("evaluation-year").value;
  const metrics = state.evaluation.metrics.filter(
    (m) => String(m.year) === year,
  );
  const names = { stan: "Stan", elo: "Elo", coin_flip: "50/50 baseline" };
  table(
    "metrics-table",
    [
      {
        key: "model",
        label: "Model",
        team: true,
        format: (v) => names[v] || v,
      },
      { key: "games", label: "Games" },
      {
        key: "log_loss",
        label: "Log loss",
        format: (v) => Number(v).toFixed(4),
      },
      {
        key: "brier",
        label: "Brier score",
        format: (v) => Number(v).toFixed(4),
      },
      { key: "accuracy", label: "Accuracy", format: percent },
      {
        key: "margin_mae_per40",
        label: "Margin MAE / 40 min",
        format: (v) => (v === null ? "—" : Number(v).toFixed(2)),
      },
    ],
    metrics,
  );
  const folds = state.evaluation.folds.filter(
    (f) => year === "pooled" || String(f.year) === year,
  );
  $("fold-diagnostics").innerHTML = folds
    .map(
      (f) =>
        `<div class="fold"><strong>${f.year}</strong><p>${f.training_games.toLocaleString()} training games · ${f.test_games} held out</p><p>Cutoff ${escapeHTML(f.cutoff)} · R-hat ${Number(f.diagnostics.max_rhat).toFixed(3)} · Min ESS ${Math.round(f.diagnostics.min_bulk_ess).toLocaleString()}</p><p>${f.diagnostics.divergences} divergences · ${f.diagnostics.max_treedepth_hits} tree-depth hits</p>${(f.warnings || []).map((w) => `<p class="status-warning">${escapeHTML(w)}</p>`).join("")}</div>`,
    )
    .join("");
  // Saved calibration bins are pooled; individual-season views explain that scope.
  renderCalibration(state.evaluation.calibration);
}
function renderCalibration(rows) {
  const colors = { stan: "#a43d17", elo: "#23634b", coin_flip: "#626866" };
  const names = { stan: "Stan", elo: "Elo", coin_flip: "50/50" };
  let svg =
    '<svg viewBox="0 0 400 330" role="img" aria-label="Pooled calibration: predicted probability versus observed win rate. The diagonal marks perfect calibration."><rect x="48" y="18" width="300" height="250" fill="#fff" stroke="#dddeda"/><path d="M48 268L348 18" stroke="#babfba" stroke-dasharray="4 4"/>';
  for (const value of [0, 0.25, 0.5, 0.75, 1])
    svg += `<text x="${48 + 300 * value}" y="288" text-anchor="middle" font-size="11" fill="#626866">${Math.round(value * 100)}%</text><text x="40" y="${272 - 250 * value}" text-anchor="end" font-size="11" fill="#626866">${Math.round(value * 100)}%</text>`;
  for (const row of rows.filter(
    (r) => r.games && r.predicted !== null && r.observed !== null,
  ))
    svg += `<circle cx="${48 + 300 * row.predicted}" cy="${268 - 250 * row.observed}" r="${Math.min(8, 3 + Math.sqrt(row.games) / 4)}" fill="${colors[row.model] || colors.coin_flip}" fill-opacity=".8"><title>${escapeHTML(names[row.model] || row.model)}: ${percent(row.predicted)} predicted, ${percent(row.observed)} observed, ${escapeHTML(row.games)} games</title></circle>`;
  svg +=
    '<text x="198" y="318" text-anchor="middle" font-size="12" fill="#626866">Predicted probability · all seasons</text><text x="13" y="145" transform="rotate(-90 13 145)" text-anchor="middle" font-size="12" fill="#626866">Observed win rate</text></svg>';
  $("calibration-chart").innerHTML =
    `<p class="help">Stan (orange) · Elo (green) · 50/50 (gray). Calibration is pooled across all seasons.</p>${svg}<details><summary>View calibration counts</summary><div class="table-scroll" id="calibration-table"></div></details>`;
  table(
    "calibration-table",
    [
      { key: "model", label: "Model", format: (v) => names[v] || v },
      { key: "games", label: "Games" },
      {
        key: "predicted",
        label: "Predicted",
        format: (v) => (v === null ? "—" : percent(v)),
      },
      {
        key: "observed",
        label: "Observed",
        format: (v) => (v === null ? "—" : percent(v)),
      },
    ],
    rows.filter((r) => r.games),
  );
}
let simulationRequest = 0;
async function loadSimulation() {
  const request = ++simulationRequest;
  state.simulation = $("simulation").value;
  if (!state.simulation) return;
  const result = await api(
    `/api/simulation?${new URLSearchParams({ run: state.simulation, draw: $("draw").value || 0 })}`,
  );
  if (request !== simulationRequest) return;
  $("simulation-result").hidden = false;
  $("simulation-empty").hidden = true;
  $("draw").max = result.draws - 1;
  download("outcomes-download", state.simulation, "outcomes.csv");
  const rounds = result.rounds.length;
  const rows = [...result.outcomes].sort(
    (a, b) => b[String(rounds - 1)] - a[String(rounds - 1)],
  );
  table(
    "outcomes-table",
    [
      { key: "seed", label: "Seed" },
      { key: "name", label: "Team", team: true },
      ...Array.from({ length: rounds }, (_, r) => ({
        key: String(r),
        label: roundName(rows.length, r),
        format: percent,
      })),
    ],
    rows,
  );
  $("bracket").innerHTML = result.rounds
    .map(
      (games, r) =>
        `<div class="round"><h3>${gameRoundName(rows.length, r)}</h3>${games.map((g) => `<div class="game"><span class="${g.first === g.winner ? "winner" : ""}">${escapeHTML(g.first)}${g.first === g.winner ? " · advances" : ""}</span><span class="${g.second === g.winner ? "winner" : ""}">${escapeHTML(g.second)}${g.second === g.winner ? " · advances" : ""}</span></div>`).join("")}</div>`,
    )
    .join("");
}
function updateJobRow(job) {
  let row = jobRows.get(job.id);
  if (!row) {
    if (!jobRows.size) $("job-history").replaceChildren();
    const node = document.createElement("div");
    node.className = "job-row";
    const title = document.createElement("strong"),
      message = document.createElement("p"),
      button = document.createElement("button");
    button.type = "button";
    button.className = "secondary";
    button.textContent = "Open saved result";
    button.dataset.jobResult = job.id;
    node.append(title, message, button);
    $("job-history").prepend(node);
    row = { node, title, message, button };
    jobRows.set(job.id, row);
  }
  const title = `${job.kind === "fit" ? "Season fit" : "Tournament simulation"} · ${job.status}`;
  if (row.title.textContent !== title) row.title.textContent = title;
  if (row.message.textContent !== job.message)
    row.message.textContent = job.message;
  row.button.hidden = !job.result;
}
async function pollJobs() {
  const previous = state.jobs;
  state.jobs = await api("/api/jobs");
  updateJobButtons();
  if (JSON.stringify(previous) === JSON.stringify(state.jobs)) return;
  const latest = state.jobs.at(-1);
  $("job-status").hidden = !latest;
  if (latest) {
    $("job-status").className =
      latest.status === "failed" ? "notice error" : "notice job";
    const message = `${latest.kind === "fit" ? "Season fit" : "Tournament simulation"} · ${latest.status}: ${latest.message}`;
    if ($("job-message").textContent !== message)
      $("job-message").textContent = message;
    $("job-open").hidden = !latest.result;
    $("job-open").dataset.jobResult = latest.id;
  }
  $("job-history").className = state.jobs.length ? "" : "empty";
  for (const job of state.jobs) updateJobRow(job);
  for (const job of state.jobs) {
    const old = previous.find((j) => j.id === job.id);
    if (
      old &&
      ["queued", "running"].includes(old.status) &&
      job.status === "complete"
    )
      await catalog(job.result.kind === "fit" ? job.result.id : "");
  }
}
async function openJob(id) {
  const job = state.jobs.find((j) => j.id === id);
  if (!job?.result) return;
  await catalog(job.result.kind === "fit" ? job.result.id : "");
  if (job.result.kind === "fit") location.hash = "ratings";
  else {
    $("simulation").value = job.result.id;
    $("draw").value = 0;
    await loadSimulation();
    location.hash = "tournament";
  }
}
$("refresh").addEventListener("click", (e) =>
  action(e.currentTarget, "Refreshing…", () => catalog()),
);
$("model").addEventListener("change", () => loadRatings().catch(showError));
$("team-search").addEventListener("input", () => {
  state.page = 0;
  renderRatings();
});
$("previous-page").addEventListener("click", () => {
  state.page--;
  renderRatings();
});
$("next-page").addEventListener("click", () => {
  state.page++;
  renderRatings();
});
$("ratings-table").addEventListener("click", (e) => {
  const button = e.target.closest("[data-sort]");
  if (!button) return;
  const key = button.dataset.sort;
  state.direction =
    state.sort === key ? -state.direction : key === "name" ? 1 : -1;
  state.sort = key;
  renderRatings();
  $("ratings-table").querySelector(`[data-sort="${key}"]`).focus();
});
$("dataset").addEventListener("change", datasetSeasons);
$("backtest-run").addEventListener("change", () =>
  loadEvaluation().catch(showError),
);
$("evaluation-year").addEventListener("change", renderEvaluation);
$("simulation").addEventListener("change", () => {
  $("draw").value = 0;
  loadSimulation().catch(showError);
});
$("draw-form").addEventListener("submit", (e) => {
  e.preventDefault();
  action(e.submitter, "Loading…", loadSimulation);
});
$("job-status").addEventListener("click", (e) => {
  const b = e.target.closest("[data-job-result]");
  if (b) action(b, "Opening…", () => openJob(b.dataset.jobResult));
});
$("job-history").addEventListener("click", (e) => {
  const b = e.target.closest("[data-job-result]");
  if (b) action(b, "Opening…", () => openJob(b.dataset.jobResult));
});
$("matchup-form").addEventListener("submit", (e) => {
  e.preventDefault();
  action(e.submitter, "Comparing…", async () => {
    const run = state.model;
    const result = await api("/api/matchup", {
      model: run,
      first: Number($("first").value),
      second: Number($("second").value),
      venue: $("venue").value,
    });
    if (run !== state.model) return;
    $("matchup-result").innerHTML =
      `<div class="matchup-teams"><div>${escapeHTML(result.first)}<strong>${percent(result.probability)}</strong></div><div>${escapeHTML(result.second)}<strong>${percent(1 - result.probability)}</strong></div></div><div class="probability-bar" aria-hidden="true"><span id="probability-fill"></span></div><p class="result-note">Expected margin: ${escapeHTML(result.first)} ${signed(result.margin)} points per 40 minutes. ${result.venue === "neutral" ? "Neutral court." : "Home-court advantage included."}</p>`;
    $("probability-fill").style.width = `${100 * result.probability}%`;
  });
});
$("fit-form").addEventListener("submit", (e) => {
  e.preventDefault();
  action(e.submitter, "Starting…", async () => {
    const [division, year] = $("season").value.split("|");
    await api("/api/fit", {
      dataset: $("dataset").value,
      division,
      year: Number(year),
      samples: Number($("samples").value),
      warmup: Number($("warmup").value),
      seed: Number($("fit-seed").value),
      legacy_home_first: $("legacy-home-first").checked,
    });
    await pollJobs();
  });
});
$("tournament-form").addEventListener("submit", (e) => {
  e.preventDefault();
  action(e.submitter, "Starting…", async () => {
    await api("/api/simulate", {
      model: state.model,
      tournament: $("tournament-file").value,
      seed: Number($("tournament-seed").value),
    });
    await pollJobs();
  });
});
window.addEventListener("hashchange", () => {
  navigate();
  window.scrollTo({ top: 0 });
});
navigate();
(async () => {
  try {
    await catalog();
    await pollJobs();
  } catch (error) {
    showError(error);
  } finally {
    setInterval(() => pollJobs().catch(showError), 3000);
  }
})();
