(() => {
  const $ = (selector) => document.querySelector(selector);
  const state = {
    adminToken: sessionStorage.getItem("smartbin_admin_token") || "",
    data: null,
    polling: false,
  };

  const els = {
    cloudDot: $("#cloudDot"), cloudText: $("#cloudText"),
    networkDevices: $("#networkDevices"), realDevices: $("#realDevices"),
    onlineDevices: $("#onlineDevices"), classifiedTotal: $("#classifiedTotal"),
    realClassified: $("#realClassified"), harbinMapValue: $("#harbinMapValue"),
    recyclableCount: $("#recyclableCount"), kitchenCount: $("#kitchenCount"),
    hazardousCount: $("#hazardousCount"), otherCount: $("#otherCount"),
    deviceBody: $("#deviceBody"), mapStage: $("#mapStage"),
    lastRefresh: $("#lastRefresh"), serverClock: $("#serverClock"),
    trendChart: $("#trendChart"), emptyChart: $("#emptyChart"),
    adminLoginBtn: $("#adminLoginBtn"), addDeviceBtn: $("#addDeviceBtn"),
    loginDialog: $("#loginDialog"), deviceDialog: $("#deviceDialog"),
    credentialDialog: $("#credentialDialog"), toast: $("#toast"),
  };

  const nf = new Intl.NumberFormat("zh-CN");

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>'"]/g, ch => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
    }[ch]));
  }

  function formatTime(iso) {
    if (!iso) return "从未上报";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "--";
    return d.toLocaleString("zh-CN", { hour12: false });
  }

  function showToast(message, isError = false) {
    els.toast.textContent = message;
    els.toast.classList.toggle("error", isError);
    els.toast.classList.add("show");
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => els.toast.classList.remove("show"), 3200);
  }

  async function api(path, options = {}, admin = false) {
    const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
    if (admin) {
      if (!state.adminToken) throw new Error("请先进行管理员登录");
      headers["X-Admin-Token"] = state.adminToken;
    }
    const response = await fetch(path, { ...options, headers });
    let body = null;
    try { body = await response.json(); } catch (_) { body = {}; }
    if (!response.ok) {
      const error = new Error(body.detail || `请求失败：${response.status}`);
      error.status = response.status;
      throw error;
    }
    return body;
  }

  function setAdminState(loggedIn) {
    els.addDeviceBtn.classList.toggle("hidden", !loggedIn);
    els.adminLoginBtn.textContent = loggedIn ? "退出管理员" : "管理员登录";
  }

  function renderMap(fakeCities) {
    els.mapStage.querySelectorAll(".map-node").forEach(node => node.remove());
    for (const city of fakeCities) {
      const node = document.createElement("div");
      node.className = "map-node";
      node.style.left = `${city.x}%`;
      node.style.top = `${city.y}%`;
      node.innerHTML = `<span>${escapeHtml(city.city)} · ${nf.format(city.devices)}</span>`;
      els.mapStage.appendChild(node);
    }
  }

  function renderDevices(devices) {
    if (!devices.length) {
      els.deviceBody.innerHTML = `<tr><td colspan="10" class="empty-row">尚未接入真实设备。管理员登录后可创建哈尔滨节点。</td></tr>`;
      return;
    }
    els.deviceBody.innerHTML = devices.map(device => {
      const c = device.counts;
      const controls = state.adminToken ? `
        <div class="controls" data-device="${escapeHtml(device.device_id)}">
          <button class="control-code" data-code="0001">可回收</button>
          <button class="control-code" data-code="0010">厨余</button>
          <button class="control-code" data-code="0100">有害</button>
          <button class="control-code" data-code="1000">其他</button>
          <button class="control-code reset" data-reset="1">清零</button>
        </div>` : `<span class="muted-control">管理员登录后可控制</span>`;
      return `<tr>
        <td><span class="status-chip ${device.online ? "online" : ""}"><i></i>${device.online ? "在线" : "离线"}</span></td>
        <td>${escapeHtml(device.device_id)}</td>
        <td>${escapeHtml(device.name)}</td>
        <td>${escapeHtml(device.city)}</td>
        <td>${nf.format(c.recyclable)}</td>
        <td>${nf.format(c.kitchen)}</td>
        <td>${nf.format(c.hazardous)}</td>
        <td>${nf.format(c.other)}</td>
        <td>${escapeHtml(formatTime(device.last_seen))}</td>
        <td>${controls}</td>
      </tr>`;
    }).join("");
  }

  function drawChart(history) {
    const canvas = els.trendChart;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(300, Math.floor(rect.width * dpr));
    canvas.height = Math.max(180, Math.floor(rect.height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const width = canvas.width / dpr;
    const height = canvas.height / dpr;
    ctx.clearRect(0, 0, width, height);

    if (!history || history.length < 2) {
      els.emptyChart.style.display = "grid";
      return;
    }
    els.emptyChart.style.display = "none";

    const points = history.slice(-80);
    const totals = points.map(p => Number(p.total || 0));
    const min = Math.min(...totals);
    const max = Math.max(...totals);
    const range = Math.max(1, max - min);
    const pad = { left: 42, right: 18, top: 18, bottom: 30 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    ctx.font = "10px Microsoft YaHei, sans-serif";
    ctx.strokeStyle = "rgba(104, 190, 230, .14)";
    ctx.fillStyle = "rgba(139, 177, 197, .7)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH * i / 4);
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
      const value = Math.round(max - range * i / 4);
      ctx.fillText(nf.format(value), 2, y + 3);
    }

    const coords = totals.map((value, index) => ({
      x: pad.left + plotW * index / (totals.length - 1),
      y: pad.top + plotH * (1 - (value - min) / range),
    }));

    const gradient = ctx.createLinearGradient(0, pad.top, 0, height - pad.bottom);
    gradient.addColorStop(0, "rgba(53, 211, 255, .28)");
    gradient.addColorStop(1, "rgba(53, 211, 255, 0)");
    ctx.beginPath();
    ctx.moveTo(coords[0].x, height - pad.bottom);
    coords.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.lineTo(coords[coords.length - 1].x, height - pad.bottom);
    ctx.closePath(); ctx.fillStyle = gradient; ctx.fill();

    ctx.beginPath();
    coords.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    ctx.strokeStyle = "#35d3ff"; ctx.lineWidth = 2; ctx.stroke();

    const last = coords[coords.length - 1];
    ctx.beginPath(); ctx.arc(last.x, last.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#4ff0b5"; ctx.fill();

    const firstTime = new Date(points[0].time).toLocaleTimeString("zh-CN", { hour12: false });
    const lastTime = new Date(points[points.length - 1].time).toLocaleTimeString("zh-CN", { hour12: false });
    ctx.fillStyle = "rgba(139, 177, 197, .75)";
    ctx.fillText(firstTime, pad.left, height - 8);
    ctx.textAlign = "right";
    ctx.fillText(lastTime, width - pad.right, height - 8);
    ctx.textAlign = "left";
  }

  function render(data) {
    state.data = data;
    const o = data.overview;
    els.networkDevices.textContent = nf.format(o.network_devices);
    els.realDevices.textContent = nf.format(o.real_devices);
    els.onlineDevices.textContent = `${nf.format(o.real_online_devices)} 在线`;
    els.classifiedTotal.textContent = nf.format(o.classified_total);
    els.realClassified.textContent = nf.format(o.real_classified_total);
    els.harbinMapValue.textContent = `真实设备 ${o.real_devices} · 在线 ${o.real_online_devices}`;

    const t = data.real_totals;
    els.recyclableCount.textContent = nf.format(t.recyclable);
    els.kitchenCount.textContent = nf.format(t.kitchen);
    els.hazardousCount.textContent = nf.format(t.hazardous);
    els.otherCount.textContent = nf.format(t.other);
    els.lastRefresh.textContent = formatTime(data.generated_at);
    els.serverClock.textContent = formatTime(data.generated_at);

    renderMap(data.fake_cities || []);
    renderDevices(data.devices || []);
    drawChart(data.history || []);
  }

  async function refreshDashboard() {
    if (state.polling) return;
    state.polling = true;
    try {
      const data = await api("/api/public/dashboard");
      render(data);
      els.cloudDot.className = "ok";
      els.cloudText.textContent = "云端服务正常";
    } catch (error) {
      els.cloudDot.className = "bad";
      els.cloudText.textContent = "云端连接异常";
      console.error(error);
    } finally {
      state.polling = false;
    }
  }

  async function sendManual(deviceId, code) {
    try {
      await api(`/api/admin/devices/${encodeURIComponent(deviceId)}/manual`, {
        method: "POST", body: JSON.stringify({ code })
      }, true);
      showToast(`已向 ${deviceId} 下发状态码 ${code}`);
      await refreshDashboard();
    } catch (error) { showToast(error.message, true); }
  }

  async function sendReset(deviceId) {
    if (!confirm(`确认清零设备 ${deviceId} 的四类垃圾计数？\n离线设备会在重新联网后执行。`)) return;
    try {
      await api(`/api/admin/devices/${encodeURIComponent(deviceId)}/reset`, {
        method: "POST", body: "{}"
      }, true);
      showToast(`清零命令已下发到 ${deviceId}`);
      await refreshDashboard();
    } catch (error) { showToast(error.message, true); }
  }

  els.deviceBody.addEventListener("click", event => {
    const button = event.target.closest("button");
    if (!button) return;
    const parent = button.closest("[data-device]");
    if (!parent) return;
    const deviceId = parent.dataset.device;
    if (button.dataset.code) sendManual(deviceId, button.dataset.code);
    if (button.dataset.reset) sendReset(deviceId);
  });

  els.adminLoginBtn.addEventListener("click", () => {
    if (state.adminToken) {
      state.adminToken = "";
      sessionStorage.removeItem("smartbin_admin_token");
      setAdminState(false);
      if (state.data) renderDevices(state.data.devices || []);
      showToast("已退出管理员模式");
    } else {
      $("#adminTokenInput").value = "";
      els.loginDialog.showModal();
    }
  });

  $("#loginForm").addEventListener("submit", async event => {
    event.preventDefault();
    const token = $("#adminTokenInput").value.trim();
    if (!token) return showToast("请输入管理员令牌", true);
    state.adminToken = token;
    try {
      await api("/api/admin/devices", {}, true);
      sessionStorage.setItem("smartbin_admin_token", token);
      setAdminState(true);
      els.loginDialog.close();
      if (state.data) renderDevices(state.data.devices || []);
      showToast("管理员登录成功");
    } catch (error) {
      state.adminToken = "";
      showToast(error.message, true);
    }
  });

  els.addDeviceBtn.addEventListener("click", () => els.deviceDialog.showModal());

  $("#deviceForm").addEventListener("submit", async event => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const valueOrNull = name => {
      const value = String(form.get(name) || "").trim();
      return value === "" ? null : value;
    };
    const payload = {
      name: valueOrNull("name"), city: valueOrNull("city"),
      device_id: valueOrNull("device_id"), note: valueOrNull("note") || "",
      longitude: valueOrNull("longitude") === null ? null : Number(valueOrNull("longitude")),
      latitude: valueOrNull("latitude") === null ? null : Number(valueOrNull("latitude")),
    };
    try {
      const result = await api("/api/admin/devices", {
        method: "POST", body: JSON.stringify(payload)
      }, true);
      els.deviceDialog.close();
      $("#createdDeviceId").value = result.device_id;
      $("#createdDeviceKey").value = result.device_key;
      $("#createdEnv").value = `SMARTBIN_SERVER=${location.origin}\nSMARTBIN_DEVICE_ID=${result.device_id}\nSMARTBIN_DEVICE_KEY=${result.device_key}\nSMARTBIN_COUNTS_FILE=/home/sunrise/smartbin/runtime/counts.json`;
      els.credentialDialog.showModal();
      await refreshDashboard();
    } catch (error) { showToast(error.message, true); }
  });

  document.querySelectorAll(".copy-btn").forEach(button => {
    button.addEventListener("click", async () => {
      const input = document.getElementById(button.dataset.copy);
      try {
        await navigator.clipboard.writeText(input.value);
        showToast("已复制到剪贴板");
      } catch (_) {
        input.select(); document.execCommand("copy"); showToast("已复制到剪贴板");
      }
    });
  });

  window.addEventListener("resize", () => state.data && drawChart(state.data.history || []));
  setAdminState(Boolean(state.adminToken));
  refreshDashboard();
  setInterval(refreshDashboard, 1000);
})();
