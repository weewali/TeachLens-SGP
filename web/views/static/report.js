// Report page: the section tabs, the moments filter, the lecture timeline (an SVG drawn at the page's width, in the
// page's colours) and the chart tooltips. Every number a tooltip shows is also printed on the page or in a table.
(function () {
  // ------------------------------------------------------------------ tabs: one section at a time
  // The links in .tabs become tabs (WAI-ARIA tabs pattern). The open tab is kept in the address (#charts), so a
  // reload or a shared link opens the same tab. Printing shows every section (style.css).
  const tabBar = document.querySelector(".tabs");
  const drawers = [];                       // charts that must draw again once their tab becomes visible
  if (tabBar) {
    const tabs = [], panels = [], names = [];
    tabBar.querySelectorAll("a[href^='#']").forEach(function (tab) {
      const name = tab.getAttribute("href").slice(1);
      const panel = document.getElementById(name);
      if (!panel) return;
      // The address says #charts, but no element keeps that id, so the browser never scrolls the page header away
      // to "jump" to the section.
      panel.id = "panel-" + name;
      tabs.push(tab); panels.push(panel); names.push(name);
    });
    tabBar.setAttribute("role", "tablist");
    const select = function (index, remember) {
      tabs.forEach(function (tab, i) {
        const on = i === index;
        tab.classList.toggle("is-active", on);
        tab.setAttribute("aria-selected", String(on));
        tab.tabIndex = on ? 0 : -1;
        panels[i].classList.toggle("is-active", on);
      });
      if (remember && window.history && history.replaceState) history.replaceState(null, "", "#" + names[index]);
      // on a narrow screen the tab bar scrolls sideways: keep the open tab fully in view
      const bar = tabBar.getBoundingClientRect(), open = tabs[index].getBoundingClientRect();
      if (open.left < bar.left) tabBar.scrollLeft -= bar.left - open.left + 16;
      else if (open.right > bar.right) tabBar.scrollLeft += open.right - bar.right + 16;
      // after scrolling down a long tab, start the next one at its top (just under the sticky tab bar)
      const top = panels[index].getBoundingClientRect().top + window.scrollY - tabBar.offsetHeight;
      if (window.scrollY > top) window.scrollTo(0, top);
    };
    tabs.forEach(function (tab, i) {
      tab.id = "tab-" + names[i];
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-controls", panels[i].id);
      panels[i].setAttribute("role", "tabpanel");
      panels[i].setAttribute("aria-labelledby", tab.id);
      tab.addEventListener("click", function (event) { event.preventDefault(); select(i, true); });
      tab.addEventListener("keydown", function (event) {
        const keys = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 };
        if (!(event.key in keys)) return;
        event.preventDefault();
        const next = (keys[event.key] + tabs.length) % tabs.length;
        select(next, true);
        tabs[next].focus();
      });
    });
    const fromAddress = function () {
      const index = names.indexOf(window.location.hash.slice(1));
      select(index < 0 ? 0 : index, false);
    };
    window.addEventListener("hashchange", fromAddress);
    fromAddress();
    // print every section with its charts drawn at full width, then go back to the open tab
    window.addEventListener("beforeprint", function () {
      document.documentElement.classList.add("printing");
      drawers.forEach(function (drawNow) { drawNow(); });
    });
    window.addEventListener("afterprint", function () { document.documentElement.classList.remove("printing"); });
  }

  // ------------------------------------------------------------------ moments filter
  const buttons = document.querySelectorAll("[data-filter]");
  const count = document.getElementById("moments-count");
  buttons.forEach(function (button) {
    button.addEventListener("click", function () {
      const kind = button.dataset.filter;
      let shown = 0;
      buttons.forEach(function (other) { other.setAttribute("aria-pressed", String(other === button)); });
      document.querySelectorAll("#moments-table tbody tr").forEach(function (row) {
        const show = kind === "all" || row.dataset.kind === kind;
        row.hidden = !show;
        if (show) shown += 1;
      });
      if (count) count.textContent = shown;
    });
  });

  // ------------------------------------------------------------------ helpers
  const NS = "http://www.w3.org/2000/svg";
  function svgEl(name, attrs, parent) {
    const node = document.createElementNS(NS, name);
    Object.keys(attrs).forEach(function (key) { node.setAttribute(key, attrs[key]); });
    if (parent) parent.appendChild(node);
    return node;
  }
  function svgText(parent, x, y, content, cls, anchor) {
    const node = svgEl("text", { x: x, y: y, class: cls, "text-anchor": anchor || "start" }, parent);
    node.textContent = content;
    return node;
  }
  function mmss(seconds) {
    const s = Math.round(seconds);
    return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
  }
  function pct(value) {
    return value === 0 ? "0%" : value < 1 ? value.toFixed(1) + "%" : Math.round(value) + "%";
  }
  // Tooltip contents are built with textContent only: labels come from data files.
  function line(parent, cls, content) {
    const div = document.createElement("div");
    div.className = cls;
    div.textContent = content;
    parent.appendChild(div);
    return div;
  }
  function placeTip(tip, figure, x, y) {
    tip.hidden = false;
    const box = figure.getBoundingClientRect();
    const width = tip.offsetWidth;
    let left = x + 14;
    if (left + width > box.width - 8) left = x - 14 - width;   // flip to the left near the right edge
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, y) + "px";
  }

  // ------------------------------------------------------------------ lecture timeline
  function timeline(figure) {
    const source = figure.querySelector(".chart-data");
    const host = figure.querySelector(".timeline-plot");
    const tip = figure.querySelector(".chart-tip");
    if (!source || !host) return;
    const data = JSON.parse(source.textContent);
    if (!data.parts.length || !(data.duration > 0)) return;
    const activities = {};
    data.activities.forEach(function (a) { activities[a.key] = a; });
    const laneNames = {};
    data.lanes.forEach(function (l) { laneNames[l.kind] = l.label; });
    let current = -1;
    let geo = null;

    function draw() {
      const width = Math.max(host.clientWidth, 260);
      const left = 42, right = 8;
      const plotW = width - left - right;
      const x = function (t) { return left + (t / data.duration) * plotW; };
      const strip = { y: 20, h: 10 };
      const plot = { y: 66, h: 160, pad: 18 };                       // pad: room above 100% for the value labels
      const y = function (v) { return plot.y + plot.pad + (1 - v / 100) * (plot.h - plot.pad); };
      const laneTop = plot.y + plot.h + 26, laneH = 38;
      const bottom = laneTop + data.lanes.length * laneH;
      const height = bottom + 22;
      geo = { left: left, plotW: plotW, x: x, y: y, top: strip.y - 4, bottom: bottom - 6 };

      host.textContent = "";
      const svg = svgEl("svg", { width: width, height: height, viewBox: "0 0 " + width + " " + height, tabindex: "0",
                                 focusable: "true", "aria-label": "Lecture timeline. Use the left and right arrow keys to read each 10-second part." }, host);

      // what the instructor was doing: one block per run of the same activity, 2px of surface between runs
      svgText(svg, left, 12, "What you were doing", "viz-label");
      let runStart = 0;
      data.parts.forEach(function (part, i) {
        const next = data.parts[i + 1];
        if (next && next.activity === part.activity) return;
        const first = data.parts[runStart];
        const x0 = x(first.start) + (runStart ? 1 : 0), x1 = x(part.end) - (next ? 1 : 0);
        svgEl("rect", { x: x0, y: strip.y, width: Math.max(x1 - x0, 1), height: strip.h, rx: 2,
                        class: "viz-act act-" + part.activity }, svg);
        runStart = i + 1;
      });

      // heads up: gridlines, area wash and line, broken where too few students were readable
      svgText(svg, left, plot.y - 10, "Students in view with their heads up", "viz-label");
      // time axis: about one label per 80px, at round steps; gridlines first so the line is drawn over them
      const steps = [10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
      const tick = steps.find(function (s) { return data.duration / s <= Math.max(2, Math.floor(plotW / 80)); }) || 3600;
      for (let t = 0; t <= data.duration + 0.01; t += tick) {
        svgEl("line", { x1: x(t), x2: x(t), y1: y(100) - 4, y2: y(0), class: "viz-grid" }, svg);
        svgText(svg, x(t), height - 4, mmss(t), "viz-tick", t === 0 ? "start" : "middle");
      }
      [0, 50, 100].forEach(function (v) {
        svgEl("line", { x1: left, x2: width - right, y1: y(v), y2: y(v), class: v === 0 ? "viz-axis" : "viz-grid" }, svg);
        svgText(svg, left - 6, y(v) + 4, v + "%", "viz-tick", "end");
      });
      const segments = [];
      let segment = [];
      data.parts.forEach(function (part) {
        if (part.heads_up === null) { if (segment.length) segments.push(segment); segment = []; return; }
        segment.push([x((part.start + part.end) / 2), y(part.heads_up)]);
      });
      if (segment.length) segments.push(segment);
      segments.forEach(function (points) {
        if (points.length === 1) {
          svgEl("circle", { cx: points[0][0], cy: points[0][1], r: 3, class: "viz-dot" }, svg);
          return;
        }
        const path = points.map(function (p, i) { return (i ? "L" : "M") + p[0].toFixed(1) + "," + p[1].toFixed(1); }).join("");
        svgEl("path", { d: path + "L" + points[points.length - 1][0].toFixed(1) + "," + y(0) + "L" + points[0][0].toFixed(1) + "," + y(0) + "Z",
                        class: "viz-area" }, svg);
        svgEl("path", { d: path, class: "viz-line" }, svg);
      });

      // label only the highest and lowest parts, not every point
      const measured = data.parts.filter(function (p) { return p.heads_up !== null; });
      if (measured.length > 1) {
        const high = measured.reduce(function (a, b) { return b.heads_up > a.heads_up ? b : a; });
        const low = measured.reduce(function (a, b) { return b.heads_up < a.heads_up ? b : a; });
        if (high.heads_up - low.heads_up >= 5) {
          [[high, "Highest " + pct(high.heads_up), -9], [low, "Lowest " + pct(low.heads_up), 18]].forEach(function (item) {
            const px = x((item[0].start + item[0].end) / 2), py = y(item[0].heads_up);
            svgEl("circle", { cx: px, cy: py, r: 4, class: "viz-dot" }, svg);
            const lx = Math.min(Math.max(px, left + 40), width - right - 40);
            const ly = item[2] > 0 && py + item[2] > y(0) - 2 ? py - 9 : py + item[2];
            svgText(svg, lx, ly, item[1], "viz-value", "middle");
          });
        }
      }

      // moments: one row per kind; each moment spans its time (at least 8px wide)
      data.lanes.forEach(function (lane, i) {
        const top = laneTop + i * laneH;
        svgText(svg, left, top + 10, lane.label + (lane.count ? " · " + lane.count : " · none"), "viz-label");
        svgEl("line", { x1: left, x2: width - right, y1: top + 24, y2: top + 24, class: "viz-grid" }, svg);
        data.moments.filter(function (m) { return m.kind === lane.kind; }).forEach(function (m) {
          const x0 = x(m.start), w = Math.max(x(m.end) - x0, 10);
          svgEl("rect", { x: Math.min(x0, width - right - w), y: top + 18, width: w, height: 12, rx: 6, class: "viz-moment" }, svg);
        });
      });

      // crosshair (moved by the pointer or the arrow keys)
      geo.cross = svgEl("line", { x1: 0, x2: 0, y1: geo.top, y2: geo.bottom, class: "viz-cross", visibility: "hidden" }, svg);
      geo.dot = svgEl("circle", { cx: 0, cy: 0, r: 4, class: "viz-dot", visibility: "hidden" }, svg);
      geo.svg = svg;

      svg.addEventListener("pointermove", function (event) {
        const box = svg.getBoundingClientRect();
        const t = ((event.clientX - box.left - left) / plotW) * data.duration;
        show(nearest(t));
      });
      svg.addEventListener("pointerleave", hide);
      svg.addEventListener("focus", function () { show(current < 0 ? 0 : current); });
      svg.addEventListener("blur", hide);
      svg.addEventListener("keydown", function (event) {
        if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
          event.preventDefault();
          const step = event.key === "ArrowRight" ? 1 : -1;
          show(Math.min(Math.max((current < 0 ? 0 : current) + step, 0), data.parts.length - 1));
        } else if (event.key === "Escape") {
          hide();
        }
      });
      if (current >= 0 && !tip.hidden) show(current);
    }

    function nearest(t) {
      let best = 0;
      data.parts.forEach(function (p, i) {
        const mid = (p.start + p.end) / 2, bestMid = (data.parts[best].start + data.parts[best].end) / 2;
        if (Math.abs(mid - t) < Math.abs(bestMid - t)) best = i;
      });
      return best;
    }

    function show(index) {
      current = index;
      const part = data.parts[index];
      const cx = geo.x((part.start + part.end) / 2);
      geo.cross.setAttribute("x1", cx);
      geo.cross.setAttribute("x2", cx);
      geo.cross.setAttribute("visibility", "visible");
      if (part.heads_up !== null) {
        geo.dot.setAttribute("cx", cx);
        geo.dot.setAttribute("cy", geo.y(part.heads_up));
        geo.dot.setAttribute("visibility", "visible");
      } else {
        geo.dot.setAttribute("visibility", "hidden");
      }
      tip.textContent = "";
      line(tip, "tip-time", mmss(part.start) + "–" + mmss(part.end));
      if (part.heads_up !== null) {
        const value = line(tip, "tip-value", pct(part.heads_up));
        const rest = document.createElement("span");
        rest.style.fontWeight = "400";
        rest.style.fontSize = ".85rem";
        rest.textContent = " heads up";
        value.appendChild(rest);
      } else {
        line(tip, "tip-row", "Too few students readable here");
      }
      const row = line(tip, "tip-row", "");
      const swatch = document.createElement("span");
      swatch.className = "swatch act-" + part.activity;
      row.appendChild(swatch);
      row.appendChild(document.createTextNode((activities[part.activity] || {}).you || part.activity));
      data.moments.filter(function (m) { return m.start < part.end && m.end > part.start; }).forEach(function (m) {
        line(tip, "tip-row", (laneNames[m.kind] || m.kind) + " · " + m.seat);
      });
      const hostBox = host.getBoundingClientRect(), figureBox = figure.getBoundingClientRect();
      placeTip(tip, figure, hostBox.left - figureBox.left + cx, hostBox.top - figureBox.top + 40);
    }

    function hide() {
      if (!geo) return;
      geo.cross.setAttribute("visibility", "hidden");
      geo.dot.setAttribute("visibility", "hidden");
      tip.hidden = true;
    }

    // Draw at the container's width. A chart in a closed tab has no width yet: it draws when its tab opens (the
    // container's width changes then). Drawing changes the height, not the width, so it does not loop.
    let pending = false, drawnWidth = 0;
    const redraw = function () {
      if (pending || host.clientWidth === 0 || host.clientWidth === drawnWidth) return;
      pending = true;
      requestAnimationFrame(function () {
        pending = false;
        if (host.clientWidth === 0) return;
        drawnWidth = host.clientWidth;
        draw();
      });
    };
    if (window.ResizeObserver) new ResizeObserver(redraw).observe(host);
    window.addEventListener("resize", redraw);
    document.querySelectorAll(".tabs a").forEach(function (tab) {
      tab.addEventListener("click", redraw);       // for browsers without ResizeObserver
    });
    drawers.push(function () { if (host.clientWidth && host.clientWidth !== drawnWidth) { drawnWidth = host.clientWidth; draw(); } });
    redraw();
  }

  // ------------------------------------------------------------------ comparison bars: tooltip on hover and focus
  function comparison(figure) {
    const tip = figure.querySelector(".chart-tip");
    if (!tip) return;
    figure.querySelectorAll("[data-tip]").forEach(function (cell) {
      const open = function () {
        tip.textContent = cell.dataset.tip;
        const box = cell.getBoundingClientRect(), figureBox = figure.getBoundingClientRect();
        placeTip(tip, figure, box.left - figureBox.left + box.width / 2, box.bottom - figureBox.top + 6);
      };
      cell.addEventListener("pointerenter", open);
      cell.addEventListener("focus", open);
      cell.addEventListener("pointerleave", function () { tip.hidden = true; });
      cell.addEventListener("blur", function () { tip.hidden = true; });
    });
  }

  document.querySelectorAll("figure.timeline").forEach(timeline);
  document.querySelectorAll("figure.compare-chart").forEach(comparison);
})();
