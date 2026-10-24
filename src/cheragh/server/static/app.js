"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    key: "",
    connected: false,
    connecting: false,
    epoch: 0,
    settings: null,
    techniques: null,
    panel: "conversation",
    controllers: new Set(),
    asking: false,
    indexing: false,
    askController: null,
    turns: [],
    toastTimer: null,
  };
  const names = {
    conversation: "Conversation",
    documents: "Documents",
    architectures: "Architectures",
  };
  const maturity = {
    stable: "Stable",
    beta: "Bêta",
    experimental: "Expérimentale",
    planned: "Planifiée",
  };
  const families = {
    indexing: "Indexation",
    retrieval: "Recherche",
    query: "Requêtes",
    augmentation: "Augmentation",
    orchestration: "Orchestration",
    structured: "Données structurées",
    multimodal: "Multimodal",
    evaluation: "Évaluation",
    governance: "Gouvernance",
  };

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  }

  function toast(message, error = false) {
    clearTimeout(state.toastTimer);
    $("toast").textContent = message;
    $("toast").className = error ? "toast error" : "toast";
    $("toast").hidden = false;
    state.toastTimer = setTimeout(() => {
      $("toast").hidden = true;
    }, 6000);
  }

  function serverStatus(text, status = "") {
    $("server-status-text").textContent = text;
    $("server-status").querySelector(".status-dot").className =
      `status-dot ${status}`;
  }

  class ApiError extends Error {
    constructor(status, message) {
      super(message);
      this.status = status;
    }
  }

  async function api(
    path,
    { body, timeout = 65000, controller = new AbortController() } = {},
  ) {
    const epoch = state.epoch;
    state.controllers.add(controller);
    const timer = setTimeout(
      () => controller.abort(new DOMException("Délai dépassé", "TimeoutError")),
      timeout,
    );
    try {
      const headers = { Accept: "application/json" };
      if (state.key) headers["X-API-Key"] = state.key;
      if (body !== undefined) headers["Content-Type"] = "application/json";
      const response = await fetch(new URL(path, window.location.href), {
        method: body === undefined ? "GET" : "POST",
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        credentials: "omit",
        cache: "no-store",
        mode: "same-origin",
        redirect: "error",
        signal: controller.signal,
      });
      const payload = await response.json();
      if (epoch !== state.epoch)
        throw new DOMException("Session terminée", "AbortError");
      if (!response.ok) {
        const messages = {
          400: "La demande n’a pas pu être traitée. Vérifiez les chemins et les paramètres.",
          401: "Clé API absente ou incorrecte. Vérifiez votre clé de connexion.",
          403: "Cette opération n’est pas autorisée par le serveur.",
          404: "Cette fonction n’est pas disponible sur ce serveur.",
          413: "La demande dépasse la taille autorisée par le serveur.",
          422: "Certains paramètres sont invalides. Vérifiez les valeurs saisies.",
          503: "Le serveur est occupé ou indisponible. Réessayez dans quelques instants.",
          504: "Le délai du serveur est dépassé. Réessayez avec une demande plus courte.",
        };
        throw new ApiError(
          response.status,
          messages[response.status] ||
            "Le serveur n’a pas pu terminer cette opération.",
        );
      }
      return payload;
    } finally {
      clearTimeout(timer);
      state.controllers.delete(controller);
    }
  }

  function errorMessage(error) {
    if (error.name === "TimeoutError")
      return "Le délai de connexion est dépassé. Vérifiez que le serveur est accessible.";
    if (error instanceof ApiError) return error.message;
    return "Impossible de joindre le serveur. Vérifiez votre connexion puis réessayez.";
  }

  function handleSessionError(error) {
    if (error instanceof ApiError && error.status === 401 && state.connected) {
      disconnect(false);
      $("connection-message").textContent = error.message;
      toast("Votre session a expiré. Reconnectez-vous.", true);
      return true;
    }
    return false;
  }

  function updateControls() {
    $("connection-form").hidden = state.connected;
    $("session-controls").hidden = !state.connected;
    $("connect-button").disabled = state.connecting;
    $("api-key").disabled = state.connecting;
    $("question").disabled = !state.connected || state.asking;
    $("ask-button").disabled = !state.connected || state.asking;
    $("cancel-button").hidden = !state.asking;
    $("clear-button").disabled =
      !state.connected || state.asking || !state.turns.length;
    $("top-k").disabled = !state.connected || state.asking;
    $("refresh-button").disabled = !state.connected;
    $("index-fields").disabled =
      !state.connected || !state.settings?.indexing_enabled || state.indexing;
    $("technique-search").disabled = !state.connected;
    $("technique-status").disabled = !state.connected;
    $("connection-copy").textContent = state.connected
      ? "Vous êtes connecté à votre espace de connaissance."
      : "Connectez votre espace pour explorer vos documents.";
    $("composer-hint").textContent = state.asking
      ? "Recherche dans vos documents…"
      : state.connected
        ? "Ctrl + Entrée pour envoyer"
        : "Connectez-vous pour commencer.";
    $("welcome-copy").textContent = state.connected
      ? "Interrogez votre index. Chaque réponse s’accompagne des passages retrouvés dans vos documents."
      : "Connectez-vous, puis posez une question sur les documents de votre index.";
    $("index-button").textContent = state.indexing
      ? "Indexation en cours…"
      : "Indexer les documents";
  }

  function clearConversation() {
    state.turns = [];
    $("messages").replaceChildren();
    $("welcome").hidden = false;
    renderSources([]);
    updateControls();
  }

  function disconnect(notify = true) {
    state.epoch += 1;
    for (const controller of state.controllers) controller.abort();
    state.controllers.clear();
    state.key = "";
    state.connected = false;
    state.connecting = false;
    state.asking = false;
    state.indexing = false;
    state.askController = null;
    state.settings = null;
    state.techniques = null;
    $("api-key").value = "";
    $("question").value = "";
    $("index-path").value = "";
    $("index-output").value = ".cheragh_index";
    $("index-output").readOnly = false;
    $("index-result").hidden = true;
    $("index-result").replaceChildren();
    $("document-count").textContent = "—";
    $("sidebar-version").textContent = "CHERAGH";
    $("top-k").value = "5";
    $("top-k").max = "50";
    $("connection-message").textContent = "";
    $("index-notice").textContent =
      "Connectez-vous pour vérifier les options d’indexation disponibles.";
    $("index-notice").className = "notice";
    $("activation-option").hidden = true;
    $("technique-grid").replaceChildren();
    $("technique-count").textContent =
      "Connectez-vous pour charger le catalogue.";
    $("technique-search").value = "";
    $("technique-status").value = "all";
    clearTimeout(state.toastTimer);
    $("toast").hidden = true;
    clearConversation();
    serverStatus("Non connecté");
    if (notify)
      toast(
        "Session fermée. La clé et les réponses ont été effacées de cet onglet.",
      );
  }

  function applySettings(settings) {
    state.settings = settings;
    const max =
      Number.isInteger(settings.max_top_k) && settings.max_top_k > 0
        ? settings.max_top_k
        : 50;
    const initial = Number.isInteger(settings.default_top_k)
      ? Math.max(1, Math.min(max, settings.default_top_k))
      : Math.min(5, max);
    $("top-k").max = String(max);
    $("top-k").value = String(initial);
    $("sidebar-version").textContent =
      `CHERAGH ${String(settings.version || "")}`;
    $("index-notice").className = settings.indexing_enabled
      ? "notice"
      : "notice warning";
    $("index-notice").textContent = settings.indexing_enabled
      ? "L’indexation est disponible dans le dossier autorisé par votre administrateur."
      : "L’indexation est désactivée sur ce serveur. Votre administrateur peut l’autoriser avec une clé API et un dossier limité.";
    $("activation-option").hidden = !settings.can_activate_index;
    $("index-activate").checked = Boolean(settings.can_activate_index);
    if (settings.index_output)
      $("index-output").value = String(settings.index_output);
    updateActivation();
  }

  function updateActivation() {
    const active = Boolean(
      state.settings?.can_activate_index && $("index-activate").checked,
    );
    $("index-output").readOnly = active;
    if (active && state.settings.index_output)
      $("index-output").value = String(state.settings.index_output);
  }

  function applyStats(stats) {
    $("document-count").textContent =
      Number.isInteger(stats.document_count) && stats.document_count >= 0
        ? new Intl.NumberFormat("fr-FR").format(stats.document_count)
        : "—";
  }

  async function refreshStatus({ quiet = false } = {}) {
    const epoch = state.epoch;
    try {
      const stats = await api("../stats");
      if (epoch !== state.epoch) return;
      applyStats(stats);
      try {
        await api("../ready", { timeout: 15000 });
        serverStatus("Serveur disponible", "online");
      } catch (error) {
        if (epoch !== state.epoch || error.name === "AbortError") return;
        serverStatus("Serveur indisponible", "offline");
      }
      if (!quiet) toast("État de l’index actualisé.");
    } catch (error) {
      if (
        epoch !== state.epoch ||
        error.name === "AbortError" ||
        handleSessionError(error)
      )
        return;
      serverStatus("Connexion interrompue", "offline");
      if (!quiet) toast(errorMessage(error), true);
    }
  }

  async function connect(event) {
    event.preventDefault();
    if (state.connecting) return;
    const key = $("api-key").value;
    disconnect(false);
    state.key = key;
    state.connecting = true;
    const epoch = state.epoch;
    updateControls();
    serverStatus("Connexion…", "busy");
    try {
      const settings = await api("api/settings", { timeout: 15000 });
      const stats = await api("../stats", { timeout: 15000 });
      if (epoch !== state.epoch) return;
      applySettings(settings);
      applyStats(stats);
      state.connected = true;
      state.connecting = false;
      updateControls();
      serverStatus("Serveur disponible", "online");
      if (state.panel === "architectures") loadTechniques();
      refreshStatus({ quiet: true });
      $("question").focus();
    } catch (error) {
      if (epoch !== state.epoch || error.name === "AbortError") return;
      state.key = "";
      state.connecting = false;
      updateControls();
      serverStatus("Non connecté", "offline");
      $("connection-message").textContent = errorMessage(error);
      $("api-key").focus();
    }
  }

  function switchPanel(panel) {
    state.panel = panel;
    for (const [name, title] of Object.entries(names)) {
      $("panel-" + name).hidden = name !== panel;
      const button = document.querySelector(`[data-panel="${name}"]`);
      button.classList.toggle("active", name === panel);
      if (name === panel) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
      if (name === panel) $("view-name").textContent = title;
    }
    if (
      panel === "architectures" &&
      state.connected &&
      state.techniques === null
    )
      loadTechniques();
  }

  function renderSources(sources) {
    const list = $("source-list");
    list.replaceChildren();
    const valid = Array.isArray(sources) ? sources : [];
    $("source-count").textContent = String(valid.length);
    if (!valid.length) {
      list.append(
        element(
          "p",
          "sources-placeholder",
          state.turns.length
            ? "Aucune source n’a été renvoyée pour cette réponse."
            : "Les passages utilisés apparaîtront ici après votre première question.",
        ),
      );
      return;
    }
    valid.forEach((source, index) => {
      const detail = element("details", "source-item");
      const title =
        source.metadata?.title ||
        source.metadata?.filename ||
        source.doc_id ||
        `Passage ${index + 1}`;
      detail.append(element("summary", "", `${index + 1}. ${String(title)}`));
      detail.append(
        element(
          "p",
          "source-preview",
          source.preview || "Aucun aperçu disponible.",
        ),
      );
      if (source.location || source.metadata?.source)
        detail.append(
          element(
            "p",
            "source-location",
            source.location || source.metadata.source,
          ),
        );
      if (typeof source.score === "number" && Number.isFinite(source.score))
        detail.append(
          element(
            "p",
            "source-score",
            `Score de recherche : ${source.score.toFixed(3)}`,
          ),
        );
      list.append(detail);
    });
  }

  function makeMessage(author, query = false) {
    const article = element(
      "article",
      query ? "message user-message" : "message assistant-message",
    );
    const heading = element("div", "message-author");
    if (!query) heading.append(element("span", "avatar", "✦"));
    heading.append(element("span", "", author));
    article.append(heading);
    return article;
  }

  function showAnswer(article, response, elapsed) {
    article.querySelector(".loading-text")?.remove();
    article.append(
      element(
        "div",
        "answer-text",
        response.answer ?? "Aucune réponse n’a été renvoyée.",
      ),
    );
    const sources = Array.isArray(response.sources) ? response.sources : [];
    const meta = element("div", "message-meta");
    meta.append(
      element(
        "span",
        "",
        `${(elapsed / 1000).toLocaleString("fr-FR", { maximumFractionDigits: 1 })} s`,
      ),
    );
    const evidence = element(
      "button",
      "",
      `${sources.length} source${sources.length === 1 ? "" : "s"}`,
    );
    evidence.type = "button";
    evidence.addEventListener("click", () => {
      renderSources(sources);
      $("source-list").scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
    meta.append(evidence);
    const copy = element("button", "", "Copier la réponse");
    copy.type = "button";
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(String(response.answer ?? ""));
        toast("Réponse copiée.");
      } catch {
        toast(
          "La copie n’est pas disponible. Sélectionnez le texte de la réponse.",
          true,
        );
      }
    });
    meta.append(copy);
    article.append(meta);
    if (Array.isArray(response.warnings) && response.warnings.length) {
      article.append(
        element(
          "p",
          "message-notice",
          "Cette réponse comporte des réserves. Vérifiez les passages sources.",
        ),
      );
      const detail = element("details", "message-notice");
      detail.append(element("summary", "", "Voir les réserves"));
      const warnings = element("ul");
      response.warnings.forEach((warning) =>
        warnings.append(element("li", "", warning)),
      );
      detail.append(warnings);
      article.append(detail);
    }
    renderSources(sources);
  }

  async function ask(event) {
    event.preventDefault();
    if (!state.connected || state.asking) return;
    const query = $("question").value.trim();
    const topK = Number($("top-k").value);
    if (!query) {
      $("question").focus();
      return;
    }
    if (!$("top-k").checkValidity()) {
      $("top-k").reportValidity();
      return;
    }
    state.asking = true;
    const epoch = state.epoch;
    const started = performance.now();
    const controller = new AbortController();
    state.askController = controller;
    $("welcome").hidden = true;
    const userMessage = makeMessage("Vous", true);
    userMessage.append(element("p", "question-text", query));
    const assistant = makeMessage("Cheragh");
    assistant.append(
      element("p", "loading-text", "Je recherche les passages pertinents…"),
    );
    $("messages").append(userMessage, assistant);
    state.turns.push({ userMessage, assistant });
    // Keep the browser workspace bounded, without persisting conversation data.
    if (state.turns.length > 20) {
      const oldest = state.turns.shift();
      oldest.userMessage.remove();
      oldest.assistant.remove();
    }
    $("question").value = "";
    updateControls();
    assistant.scrollIntoView({ behavior: "smooth", block: "nearest" });
    try {
      const response = await api("../ask", {
        body: { query, top_k: topK },
        controller,
        timeout:
          (Number(state.settings.request_timeout_seconds) || 60) * 1000 + 5000,
      });
      if (epoch !== state.epoch) return;
      showAnswer(assistant, response, performance.now() - started);
    } catch (error) {
      if (epoch !== state.epoch || handleSessionError(error)) return;
      assistant.querySelector(".loading-text")?.remove();
      const message =
        error.name === "AbortError"
          ? "Requête annulée dans cet onglet. Le serveur peut encore terminer son traitement."
          : errorMessage(error);
      assistant.append(element("p", "answer-text message-error", message));
      renderSources([]);
    } finally {
      if (epoch === state.epoch) {
        state.asking = false;
        state.askController = null;
        updateControls();
        $("question").focus();
      }
    }
  }

  async function indexDocuments(event) {
    event.preventDefault();
    if (!state.connected || !state.settings?.indexing_enabled || state.indexing)
      return;
    const size = Number($("chunk-size").value);
    const overlap = Number($("chunk-overlap").value);
    $("chunk-overlap").setCustomValidity(
      overlap >= size
        ? "Le chevauchement doit être inférieur à la taille des fragments."
        : "",
    );
    if (!$("index-form").reportValidity()) return;
    const body = {
      path: $("index-path").value.trim(),
      output: $("index-output").value.trim(),
      chunk_size: size,
      chunk_overlap: overlap,
      incremental: $("index-incremental").checked,
      activate: Boolean(
        state.settings.can_activate_index && $("index-activate").checked,
      ),
    };
    if (!body.path || !body.output) {
      toast("Indiquez les chemins des documents et de l’index.", true);
      return;
    }
    state.indexing = true;
    const epoch = state.epoch;
    updateControls();
    const result = $("index-result");
    result.hidden = false;
    result.className = "index-result";
    result.replaceChildren(
      element(
        "p",
        "loading-text",
        "Indexation en cours. Cette opération peut prendre quelques minutes…",
      ),
    );
    try {
      const response = await api("../index", {
        body,
        timeout:
          (Number(state.settings.index_timeout_seconds) || 900) * 1000 + 5000,
      });
      if (epoch !== state.epoch) return;
      result.replaceChildren(
        element(
          "strong",
          "",
          response.activated
            ? "Votre index est prêt à être interrogé."
            : "L’index a été créé.",
        ),
      );
      if (Number.isInteger(response.indexed_documents))
        result.append(
          element(
            "p",
            "",
            `${response.indexed_documents.toLocaleString("fr-FR")} fragments indexés.`,
          ),
        );
      result.append(
        element(
          "p",
          "",
          response.activated
            ? "L’index actif a été actualisé. Revenez dans Conversation pour explorer vos documents."
            : "Le serveur conserve son index actuel. Redémarrez-le avec le nouvel index pour l’utiliser.",
        ),
      );
      const detail = element("details");
      detail.append(element("summary", "", "Détails de l’indexation"));
      detail.append(element("pre", "", JSON.stringify(response, null, 2)));
      result.append(detail);
      if (response.activated) await refreshStatus({ quiet: true });
    } catch (error) {
      if (
        epoch !== state.epoch ||
        error.name === "AbortError" ||
        handleSessionError(error)
      )
        return;
      result.className = "index-result error";
      const message =
        error instanceof ApiError && error.status === 504
          ? "Le délai d’indexation est dépassé. Les fichiers peuvent encore être traités sur le serveur ; vérifiez l’index avant de relancer."
          : errorMessage(error);
      result.replaceChildren(element("p", "", message));
    } finally {
      if (epoch === state.epoch) {
        state.indexing = false;
        updateControls();
      }
    }
  }

  async function loadTechniques() {
    const epoch = state.epoch;
    $("technique-count").textContent = "Chargement du catalogue…";
    try {
      const response = await api("api/techniques", { timeout: 15000 });
      if (epoch !== state.epoch) return;
      state.techniques = Array.isArray(response) ? response : [];
      renderTechniques();
    } catch (error) {
      if (
        epoch !== state.epoch ||
        error.name === "AbortError" ||
        handleSessionError(error)
      )
        return;
      $("technique-count").textContent = "Catalogue indisponible.";
      toast(errorMessage(error), true);
    }
  }

  function renderTechniques() {
    if (!state.techniques) return;
    const search = $("technique-search").value.trim().toLocaleLowerCase("fr");
    const status = $("technique-status").value;
    const filtered = state.techniques.filter(
      (technique) =>
        (status === "all" || technique.status === status) &&
        [
          technique.name,
          technique.id,
          technique.family,
          families[technique.family],
          technique.summary,
        ].some((value) =>
          String(value || "")
            .toLocaleLowerCase("fr")
            .includes(search),
        ),
    );
    $("technique-count").textContent =
      `${filtered.length} architecture${filtered.length === 1 ? "" : "s"} sur ${state.techniques.length}.`;
    const grid = $("technique-grid");
    grid.replaceChildren();
    if (!filtered.length) {
      grid.append(
        element(
          "p",
          "catalog-empty",
          "Aucune architecture ne correspond à votre recherche.",
        ),
      );
      return;
    }
    for (const technique of filtered) {
      const article = element("article", "technique-card");
      const topline = element("div", "technique-topline");
      topline.append(
        element(
          "span",
          "technique-family",
          families[technique.family] || technique.family,
        ),
      );
      const knownStatus = Object.hasOwn(maturity, technique.status)
        ? technique.status
        : "planned";
      topline.append(
        element("span", `maturity ${knownStatus}`, maturity[knownStatus]),
      );
      article.append(
        topline,
        element("h2", "", technique.name),
        element("p", "", technique.summary),
      );
      const details = element("details");
      details.append(element("summary", "", "Détails et limites"));
      details.append(
        element(
          "p",
          "",
          technique.available
            ? "Implémentation disponible dans le package."
            : "Implémentation non disponible.",
        ),
      );
      if (
        Array.isArray(technique.limitations) &&
        technique.limitations.length
      ) {
        const list = element("ul");
        technique.limitations.forEach((limit) =>
          list.append(element("li", "", limit)),
        );
        details.append(list);
      } else
        details.append(
          element(
            "p",
            "",
            "Évaluez cette technique avec vos modèles, vos données et votre charge.",
          ),
        );
      article.append(details);
      grid.append(article);
    }
  }

  $("connection-form").addEventListener("submit", connect);
  $("disconnect-button").addEventListener("click", () => disconnect());
  $("ask-form").addEventListener("submit", ask);
  $("cancel-button").addEventListener("click", () =>
    state.askController?.abort(),
  );
  $("clear-button").addEventListener("click", () => {
    clearConversation();
    $("question").focus();
  });
  $("refresh-button").addEventListener("click", () => refreshStatus());
  $("index-form").addEventListener("submit", indexDocuments);
  $("index-activate").addEventListener("change", updateActivation);
  $("chunk-overlap").addEventListener("input", () =>
    $("chunk-overlap").setCustomValidity(""),
  );
  $("chunk-size").addEventListener("input", () =>
    $("chunk-overlap").setCustomValidity(""),
  );
  $("technique-search").addEventListener("input", renderTechniques);
  $("technique-status").addEventListener("change", renderTechniques);
  $("question").addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      $("ask-form").requestSubmit();
    }
  });
  document
    .querySelectorAll("[data-panel]")
    .forEach((button) =>
      button.addEventListener("click", () => switchPanel(button.dataset.panel)),
    );
  document.querySelectorAll("[data-question]").forEach((button) =>
    button.addEventListener("click", () => {
      if (!state.connected) {
        $("api-key").focus();
        toast("Connectez-vous pour poser votre première question.");
        return;
      }
      $("question").value = button.dataset.question;
      $("question").focus();
    }),
  );
  // BFCache must not restore an authenticated workspace after navigating away.
  window.addEventListener("pagehide", () => disconnect(false));
  updateControls();
})();
