/* Live kernel SVM: an RBF support vector machine fitted in the browser with
   simplified SMO. Click the field to add points; the boundary refits and the
   support vectors are ringed. data-boundary="points" draws dots,
   data-boundary="splats" draws each sample as an anisotropic Gaussian. */
(function () {
  "use strict";
  var cv = document.querySelector("canvas[data-boundary]");
  if (!cv) return;
  var mode = cv.getAttribute("data-boundary");
  var ctx = cv.getContext("2d");
  var statusEl = document.querySelector("[data-status]");
  var chipA = document.querySelector('[data-class="1"]');
  var chipB = document.querySelector('[data-class="-1"]');
  var resetBtn = document.querySelector("[data-reset]");
  var JA = (document.documentElement.lang || "").indexOf("ja") === 0;

  var css = getComputedStyle(document.documentElement);
  var COL = {
    ink: css.getPropertyValue("--ink").trim() || "#18202b",
    a: css.getPropertyValue("--cobalt").trim() || "#2e5aac",
    b: css.getPropertyValue("--plum").trim() || "#8b2f5e",
    sv: css.getPropertyValue("--mustard").trim() || "#c99a12"
  };
  function rgb(hex) {
    var h = hex.replace("#", "");
    return [parseInt(h.substr(0, 2), 16), parseInt(h.substr(2, 2), 16), parseInt(h.substr(4, 2), 16)];
  }
  var RA = rgb(COL.a), RB = rgb(COL.b);

  // Seeded RNG so the opening picture is the same for every visitor.
  function rng(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  var rand = rng(27156);
  function gauss() { var u = rand() || 1e-9, v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  var pts = [], alpha = [], b = 0, current = 1, W = 0, H = 0, AR = 2.5;
  var C = 4, GAMMA = 7, MAXP = 200;

  function makePoint(x, y, c) {
    return { x: x, y: y, c: c, rot: rand() * Math.PI, sx: 0.6 + rand() * 0.9, sy: 0.35 + rand() * 0.45 };
  }
  function seedData() {
    pts = [];
    for (var i = 0; i < 34; i++) {
      var t = Math.PI * (i / 33);
      var xa = Math.cos(t) + 0.09 * gauss(), ya = Math.sin(t) + 0.09 * gauss();
      var xb = 1 - Math.cos(t) + 0.09 * gauss(), yb = 0.45 - Math.sin(t) + 0.09 * gauss();
      pts.push(makePoint((xa + 1.35) / 3.7, (1.25 - ya) / 1.95, 1));
      pts.push(makePoint((xb + 1.35) / 3.7, (1.25 - yb) / 1.95, -1));
    }
  }

  function k(p, q) {
    var dx = (p.x - q.x) * AR, dy = p.y - q.y;
    return Math.exp(-GAMMA * (dx * dx + dy * dy));
  }

  function fit() {
    var n = pts.length, K = [], i, j;
    for (i = 0; i < n; i++) { K[i] = new Float64Array(n); }
    for (i = 0; i < n; i++) for (j = i; j < n; j++) { K[i][j] = K[j][i] = k(pts[i], pts[j]); }
    alpha = new Float64Array(n); b = 0;
    var tol = 1e-3, passes = 0, iter = 0, r = rng(n * 7919);
    function f(i) { var s = b; for (var t = 0; t < n; t++) if (alpha[t]) s += alpha[t] * pts[t].c * K[t][i]; return s; }
    while (passes < 8 && iter < 400) {
      var changed = 0; iter++;
      for (i = 0; i < n; i++) {
        var yi = pts[i].c, Ei = f(i) - yi;
        if ((yi * Ei < -tol && alpha[i] < C) || (yi * Ei > tol && alpha[i] > 0)) {
          j = Math.floor(r() * (n - 1)); if (j >= i) j++;
          var yj = pts[j].c, Ej = f(j) - yj, ai = alpha[i], aj = alpha[j], L, Hh;
          if (yi !== yj) { L = Math.max(0, aj - ai); Hh = Math.min(C, C + aj - ai); }
          else { L = Math.max(0, ai + aj - C); Hh = Math.min(C, ai + aj); }
          if (L === Hh) continue;
          var eta = 2 * K[i][j] - K[i][i] - K[j][j];
          if (eta >= 0) continue;
          var ajn = aj - (yj * (Ei - Ej)) / eta;
          ajn = Math.min(Hh, Math.max(L, ajn));
          if (Math.abs(ajn - aj) < 1e-5) continue;
          var ain = ai + yi * yj * (aj - ajn);
          var b1 = b - Ei - yi * (ain - ai) * K[i][i] - yj * (ajn - aj) * K[i][j];
          var b2 = b - Ej - yi * (ain - ai) * K[i][j] - yj * (ajn - aj) * K[j][j];
          alpha[i] = ain; alpha[j] = ajn;
          b = (ain > 0 && ain < C) ? b1 : (ajn > 0 && ajn < C) ? b2 : (b1 + b2) / 2;
          changed++;
        }
      }
      passes = changed === 0 ? passes + 1 : 0;
    }
  }

  function decision(x, y) {
    var s = b, q = { x: x, y: y };
    for (var t = 0; t < pts.length; t++) if (alpha[t] > 1e-6) s += alpha[t] * pts[t].c * k(pts[t], q);
    return s;
  }

  function contour(grid, cols, rows, step, level) {
    ctx.beginPath();
    for (var r = 0; r < rows; r++) for (var c = 0; c < cols; c++) {
      var v0 = grid[r][c] - level, v1 = grid[r][c + 1] - level, v2 = grid[r + 1][c + 1] - level, v3 = grid[r + 1][c] - level;
      var x = c * step, y = r * step, P = [];
      if ((v0 > 0) !== (v1 > 0)) P.push([x + step * v0 / (v0 - v1), y]);
      if ((v1 > 0) !== (v2 > 0)) P.push([x + step, y + step * v1 / (v1 - v2)]);
      if ((v3 > 0) !== (v2 > 0)) P.push([x + step * v3 / (v3 - v2), y + step]);
      if ((v0 > 0) !== (v3 > 0)) P.push([x, y + step * v0 / (v0 - v3)]);
      for (var p = 0; p + 1 < P.length; p += 2) { ctx.moveTo(P[p][0], P[p][1]); ctx.lineTo(P[p + 1][0], P[p + 1][1]); }
    }
    ctx.stroke();
  }

  function draw() {
    var step = 7, cols = Math.ceil(W / step), rows = Math.ceil(H / step), grid = [], r, c;
    for (r = 0; r <= rows; r++) { grid[r] = new Float64Array(cols + 1); for (c = 0; c <= cols; c++) grid[r][c] = decision(c * step / W, r * step / H); }
    ctx.clearRect(0, 0, W, H);
    for (r = 0; r < rows; r++) for (c = 0; c < cols; c++) {
      var v = (grid[r][c] + grid[r][c + 1] + grid[r + 1][c] + grid[r + 1][c + 1]) / 4;
      var base = v > 0 ? RA : RB, a = Math.abs(v) < 1 ? 0.16 : 0.07;
      ctx.fillStyle = "rgba(" + base[0] + "," + base[1] + "," + base[2] + "," + a + ")";
      ctx.fillRect(c * step, r * step, step, step);
    }
    ctx.strokeStyle = COL.ink; ctx.globalAlpha = 0.45; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    contour(grid, cols, rows, step, 1); contour(grid, cols, rows, step, -1);
    ctx.setLineDash([]); ctx.globalAlpha = 1; ctx.lineWidth = 2;
    contour(grid, cols, rows, step, 0);

    var nsv = 0;
    for (var i = 0; i < pts.length; i++) {
      var p = pts[i], px = p.x * W, py = p.y * H, col = p.c > 0 ? RA : RB;
      if (mode === "splats") {
        ctx.save(); ctx.translate(px, py); ctx.rotate(p.rot); ctx.scale(16 * p.sx, 16 * p.sy);
        var g = ctx.createRadialGradient(0, 0, 0, 0, 0, 1);
        g.addColorStop(0, "rgba(" + col[0] + "," + col[1] + "," + col[2] + ",0.7)");
        g.addColorStop(1, "rgba(" + col[0] + "," + col[1] + "," + col[2] + ",0)");
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(0, 0, 1, 0, 2 * Math.PI); ctx.fill(); ctx.restore();
      } else {
        ctx.fillStyle = p.c > 0 ? COL.a : COL.b;
        ctx.beginPath(); ctx.arc(px, py, 4.5, 0, 2 * Math.PI); ctx.fill();
      }
      if (alpha[i] > 1e-6) {
        nsv++;
        ctx.strokeStyle = COL.sv; ctx.lineWidth = 2.25;
        ctx.beginPath(); ctx.arc(px, py, 9.5, 0, 2 * Math.PI); ctx.stroke();
      }
    }
    if (statusEl) {
      statusEl.innerHTML = JA
        ? (mode === "splats" ? pts.length + "サンプル中 " : pts.length + "点中 ") + "<strong>" + nsv + "</strong> " +
          (mode === "splats" ? "個" : "点") + "がこの境界を支えています。"
        : "<strong>" + nsv + "</strong> of " + pts.length + " " +
          (mode === "splats" ? "samples" : "points") + " hold up this boundary.";
    }
  }

  function resize() {
    var dpr = window.devicePixelRatio || 1, rect = cv.getBoundingClientRect();
    W = rect.width; H = rect.height; AR = W / H;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    fit(); draw();
  }

  function setClass(c) {
    current = c;
    if (chipA) chipA.setAttribute("aria-pressed", String(c === 1));
    if (chipB) chipB.setAttribute("aria-pressed", String(c === -1));
  }
  cv.addEventListener("pointerdown", function (e) {
    var rect = cv.getBoundingClientRect();
    if (pts.length >= MAXP) pts.shift();
    pts.push(makePoint((e.clientX - rect.left) / rect.width, (e.clientY - rect.top) / rect.height, e.shiftKey ? -current : current));
    fit(); draw();
  });
  if (chipA) chipA.addEventListener("click", function () { setClass(1); });
  if (chipB) chipB.addEventListener("click", function () { setClass(-1); });
  if (resetBtn) resetBtn.addEventListener("click", function () { rand = rng(27156); seedData(); fit(); draw(); });

  var t;
  window.addEventListener("resize", function () { clearTimeout(t); t = setTimeout(resize, 120); });
  seedData(); setClass(1); resize();
})();

/* BibTeX copy */
(function () {
  var JA = (document.documentElement.lang || "").indexOf("ja") === 0;
  var T = JA ? { copy: "BibTeXをコピー", done: "コピーしました", fail: "テキストを選択してコピーしてください" }
             : { copy: "Copy BibTeX", done: "Copied", fail: "Select the text to copy" };
  var btns = document.querySelectorAll("[data-copy]");
  Array.prototype.forEach.call(btns, function (btn) {
    btn.addEventListener("click", function () {
      var src = document.getElementById(btn.getAttribute("data-copy"));
      if (!src || !navigator.clipboard) { btn.textContent = T.fail; return; }
      navigator.clipboard.writeText(src.textContent).then(function () {
        btn.textContent = T.done;
        setTimeout(function () { btn.textContent = T.copy; }, 1800);
      }, function () { btn.textContent = T.fail; });
    });
  });
})();

/* Off the production host (local preview), point directory links at index.html. */
(function () {
  if (/(^|\.)ohsaworks\.com$|github\.io$/.test(location.hostname)) return;
  Array.prototype.forEach.call(document.querySelectorAll("a[href]"), function (a) {
    var h = a.getAttribute("href");
    if (!/^[a-z]+:|^#|^\//i.test(h) && /\/$/.test(h)) a.setAttribute("href", h + "index.html");
  });
})();
