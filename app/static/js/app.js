/**
 * CatBreedID - Application Logic (tuần 2: nối giao diện với dịch vụ FastAPI)
 * Ảnh người dùng tải lên  -> gọi POST /predict của dịch vụ (mô hình tạm).
 * Ảnh mẫu trong img/ và ảnh tải lên đều được gửi đến dịch vụ.
 * Không cần sửa index.html: phần gọi dịch vụ đã nằm trong file này.
 */

// DOM Selector Helper
const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const fmt = (v, d = 2) => Number(v).toFixed(d).replace(".", ",");

// ---------------------------------------------------------------------------
// Cấu hình dịch vụ
// ---------------------------------------------------------------------------
const API_URL = window.CATBREED_API_URL || "http://127.0.0.1:8000";
const MAX_BYTES = 8 * 1024 * 1024; // khớp với giới hạn của api/main.py

/** Gửi ảnh lên dịch vụ, trả về JSON (định dạng xem api/temp_model.py). */
async function analyzeImage(file) {
  const form = new FormData();
  form.append("file", file);
  let res;
  try {
    res = await fetch(API_URL + "/predict", { method: "POST", body: form });
  } catch (e) {
    throw new Error(
      "Không kết nối được dịch vụ. Tại thư mục dự án, chạy: .\\.venv\\Scripts\\python.exe -m uvicorn app.api.main:app --port 8000",
    );
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Dịch vụ trả lỗi " + res.status);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// SVG Icons
// ---------------------------------------------------------------------------
const CHECK =
  '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"></path></svg>';
const WARN =
  '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"></circle><path d="M12 7.5v5.5M12 16.5v.5"></path></svg>';

// Mô tả ngắn từng giống. Mới có Abyssinian; nhóm điền thêm các giống còn lại
// (giống chưa có mô tả sẽ không hiện đoạn mô tả).
const BREED_DESC = {
  Abyssinian:
    "Giống mèo lông ngắn cổ xưa với bộ lông vằn agouti đặc trưng, đôi tai to linh hoạt và dáng vẻ nhanh nhẹn, quý phái.",
};

// ---------------------------------------------------------------------------
// Application State
let imgURL = null;
let currentFile = null; // File người dùng tải lên (null khi đang dùng ảnh mẫu)

/** Apply image preview background to element */
function paint(el) {
  if (!el) return;
  if (imgURL) {
    el.style.backgroundImage = 'url("' + imgURL + '")';
    el.classList.add("has");
  } else {
    el.style.backgroundImage = "";
    el.classList.remove("has");
  }
}

/** Select a sample image from the pre-provided dataset */
function selectSample(url, name) {
  if (imgURL && imgURL.startsWith("blob:")) {
    URL.revokeObjectURL(imgURL);
  }
  imgURL = url;
  currentFile = null;

  paint($("pvPhoto"));
  $("fname").textContent = name;
  $("drop").hidden = true;
  $("prev").hidden = false;
  $("status").textContent = "";
  $("go").disabled = false;
  $("go").textContent = "Phân tích ảnh";

  $("prev").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/** Handle new image file upload (file picker or drag-and-drop) */
function setImage(file) {
  if (!file || !file.type.startsWith("image/")) return;
  if (imgURL && imgURL.startsWith("blob:")) {
    URL.revokeObjectURL(imgURL);
  }
  imgURL = URL.createObjectURL(file);
  currentFile = file; // kết quả sẽ do dịch vụ trả về, không đoán theo tên file nữa

  paint($("pvPhoto"));
  $("fname").textContent = file.name;
  $("drop").hidden = true;
  $("prev").hidden = false;
  $("go").textContent = "Phân tích ảnh";
  if (file.size > MAX_BYTES) {
    $("status").textContent = "Ảnh quá lớn (tối đa 8 MB). Hãy chọn ảnh khác.";
    $("go").disabled = true;
  } else {
    $("status").textContent = "";
    $("go").disabled = false;
  }

  $("prev").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/** Reset application state to initial upload screen */
function reset() {
  if (imgURL && imgURL.startsWith("blob:")) {
    URL.revokeObjectURL(imgURL);
  }
  imgURL = null;
  currentFile = null;
  $("result").hidden = true;
  $("home").hidden = false;
  $("prev").hidden = true;
  $("drop").hidden = false;
  $("file").value = "";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/** Trích xuất phương pháp OOD tối ưu nhất (Mahalanobis Feature Space) */
function getBestOOD(r) {
  return r.ood || null;
}

/** Render kết quả: Tập trung Dự đoán, Kiểm định OOD tối ưu tích hợp bên dưới & Grad-CAM */
function render(r) {
  const unverified = r.status === "unverified";
  const isOk = r.status === "accepted" || unverified;
  const top = r.top5 && r.top5[0] ? r.top5[0] : { label: "Chưa rõ", prob: 0 };
  const pct = Math.round(top.prob * 100);
  const notice =
    r.notice ||
    (r.is_mock
      ? "Kết quả giả lập từ mô hình tạm, chưa phải kết quả của mô hình thật."
      : "");
  const noticeHtml = notice
    ? '<p class="note notice-pill">' + esc(notice) + "</p>"
    : "";

  // Lấy thông số OOD tối ưu nhất (Mahalanobis Feature Space)
  const bestOod = getBestOOD(r);
  const cleanMethod = (bestOod?.method || "")
    .replace(/\s*\([^)]*\)/g, "")
    .trim();
  const scoreFormatted = bestOod
    ? (bestOod.score > 0 ? "+" : "") + fmt(bestOod.score)
    : "—";
  const threshFormatted = bestOod ? fmt(bestOod.threshold) : "—";
  const latencyFormatted = r.latency_ms ? r.latency_ms + " ms" : "—";

  // HTML thẻ kiểm định OOD tốt nhất tích hợp bên dưới dự đoán
  const rawScores = Array.isArray(r.ood_all) && !r.is_mock ? r.ood_all : [];
  const rawScoresHtml = rawScores.length
    ? `
    <table class="ood-score-table"><thead><tr><th>Phương pháp</th><th>Điểm</th></tr></thead><tbody>
      ${rawScores.map((m) => `<tr><td>${esc(m.method)}</td><td>${Number.isFinite(Number(m.score)) ? esc(fmt(m.score, 4)) : "—"}</td></tr>`).join("")}
    </tbody></table>`
    : "";
  const trustCardHtml =
    unverified || !bestOod
      ? `
        <div class="ood-trust-card">
          <div class="ood-trust-title">Kiểm tra ảnh ngoài phân phối</div>
          <p class="ood-trust-desc">Chưa có ngưỡng OOD được hiệu chỉnh cho checkpoint này. Kết quả phía trên là giống có điểm cao nhất trong 12 lớp; hệ thống chưa xác định ảnh có thuộc các lớp đó hay không.</p>
          ${rawScoresHtml}
          <p class="note">Thời gian suy luận: ${esc(latencyFormatted)}</p>
        </div>`
      : `
        <div class="ood-trust-card ${isOk ? "is-ok" : "is-bad"}">
            <div class="ood-trust-head">
                <div class="ood-trust-title">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                    </svg>
                    <span>Kiểm định ngoại lai (${esc(cleanMethod)})</span>
                </div>
                <span class="ood-badge ${isOk ? "badge-ok" : "badge-bad"}">
                    ${isOk ? CHECK + "Đạt chuẩn In-Distribution" : WARN + "Cảnh báo Out-of-Distribution"}
                </span>
            </div>
            <div class="ood-metrics-grid">
                <div class="ood-metric-item">
                    <span class="m-lbl">Điểm đo lường</span>
                    <span class="m-val ${isOk ? "txt-ok" : "txt-bad"}">${scoreFormatted}</span>
                </div>
                <div class="ood-metric-item">
                    <span class="m-lbl">Ngưỡng an toàn (τ)</span>
                    <span class="m-val">${threshFormatted}</span>
                </div>
                <div class="ood-metric-item">
                    <span class="m-lbl">Thời gian suy luận</span>
                    <span class="m-val">${latencyFormatted}</span>
                </div>
                <div class="ood-metric-item">
                    <span class="m-lbl">Trạng thái lọc</span>
                    <span class="m-val ${isOk ? "txt-ok" : "txt-bad"}">${isOk ? "Cho phép nhận diện" : "Kích hoạt từ chối"}</span>
                </div>
            </div>
            <p class="ood-trust-desc">
                ${
                  isOk
                    ? "Điểm của phương pháp chính đạt ngưỡng chấp nhận đã cấu hình."
                    : "Điểm của phương pháp chính thấp hơn ngưỡng chấp nhận đã cấu hình."
                }
            </p>
        </div>
    `;

  let html;
  if (isOk) {
    const desc = BREED_DESC[r.top_label];
    html =
      '<div class="res-head">' +
      '<span class="res-eyebrow">Kết quả phân loại giống</span>' +
      '<div class="res-title-row">' +
      "<h1>" +
      esc(r.top_label) +
      "</h1>" +
      '<span class="chip">' +
      CHECK +
      (unverified ? "Chưa kiểm định OOD" : "Đã qua ngưỡng OOD") +
      "</span>" +
      "</div>" +
      '<div class="meter-wrap">' +
      '<div class="meter-info"><span>Độ tự tin Softmax</span><b>' +
      pct +
      "%</b></div>" +
      '<div class="meter"><div class="mt"><i style="width:0%"></i></div></div>' +
      "</div>" +
      (desc ? '<p class="desc">' + esc(desc) + "</p>" : "") +
      noticeHtml +
      trustCardHtml +
      '<div class="res-actions">' +
      '<button class="btn pri" id="again"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path><path d="M3 3v5h5"></path></svg> Chọn ảnh khác</button>' +
      "</div>" +
      "</div>";
  } else {
    const desc =
      r.desc ||
      "Ảnh này khác với những ảnh mèo hệ thống đã học. Có thể trong ảnh không có mèo, hoặc đây là vật nuôi thuộc loài khác.";
    html =
      '<div class="res-head">' +
      '<span class="res-eyebrow">Kết quả kiểm định tập mở</span>' +
      '<div class="res-title-row">' +
      "<h1>Mình chưa chắc đây là giống nào</h1>" +
      '<span class="chip no">' +
      WARN +
      "Chưa xác định</span>" +
      "</div>" +
      '<div class="meter-wrap">' +
      '<div class="meter-info"><span>Độ tin cậy tối đa</span><b>' +
      pct +
      "%</b></div>" +
      '<div class="meter"><div class="mt"><i style="width:0%;background:#F59E0B"></i></div></div>' +
      "</div>" +
      '<p class="desc">' +
      esc(desc) +
      "</p>" +
      noticeHtml +
      trustCardHtml +
      '<div class="res-actions">' +
      '<button class="btn pri" id="again"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path><path d="M3 3v5h5"></path></svg> Chọn ảnh khác</button>' +
      "</div>" +
      "</div>";
  }
  $("res").innerHTML = html;

  // Trigger smooth confidence meter animation
  requestAnimationFrame(() => {
    const bar = $("res").querySelector(".mt i");
    if (bar) {
      setTimeout(() => {
        bar.style.width = pct + "%";
      }, 60);
    }
  });

  // Các khả năng khác: ok bỏ giống đứng đầu, từ chối hiện cả giống gần nhất
  const others = isOk ? r.top5.slice(1, 5) : r.top5.slice(0, 4);
  if ($("tips")) $("tips").hidden = isOk;
  const hasHeatmap =
    typeof r.heatmap === "string" &&
    typeof r.cam_input === "string" &&
    /^[A-Za-z0-9+/=]+$/.test(r.heatmap) &&
    /^[A-Za-z0-9+/=]+$/.test(r.cam_input);
  $("why").hidden = !hasHeatmap;
  if (hasHeatmap) {
    $("hm").style.backgroundImage =
      'url("data:image/png;base64,' + r.cam_input + '")';
    $("ov").style.backgroundImage =
      'url("data:image/png;base64,' + r.heatmap + '")';
  } else {
    $("ov").style.backgroundImage = "";
  }
  $("othTitle").textContent = isOk
    ? "Các khả năng khác"
    : "Máy đang phân vân giữa";
  $("othNote").hidden = isOk;
  $("oth").className = "others" + (isOk ? "" : " dim");
  $("oth").innerHTML = others
    .map((o) => {
      const p = Math.round(o.prob * 100);
      return (
        '<div class="o"><span>' +
        esc(o.label) +
        '</span><div class="mt"><i style="width:' +
        p +
        '%"></i></div><b>' +
        p +
        "%</b></div>"
      );
    })
    .join("");
}

/** Switch from home to result view */
function showResult(r) {
  $("home").hidden = true;
  $("result").hidden = false;
  paint($("resPhoto"));
  paint($("hm"));
  applyOv();
  render(r);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/** Update Grad-CAM heat map opacity */
function applyOv() {
  $("ov").style.opacity = $("showOv").checked ? $("opa").value / 100 : 0;
}

// Event Listeners: Sample Buttons
document.querySelectorAll(".sample-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const url = btn.dataset.img;
    const name = btn.dataset.name;
    selectSample(url, name);
  });
});

// Event Listeners: File Upload & Preview Actions
$("pick").onclick = () => $("file").click();
$("file").onchange = (e) => setImage(e.target.files[0]);
$("swap").onclick = () => {
  $("file").value = "";
  $("file").click();
};
$("go").onclick = async () => {
  $("go").disabled = true;
  $("go").textContent = "Đang phân tích…";
  $("status").textContent = "Đang gửi ảnh đến dịch vụ để phân tích…";
  try {
    let result;
    if (currentFile) {
      result = await analyzeImage(currentFile);
    } else {
      const sampleResponse = await fetch(imgURL);
      if (!sampleResponse.ok) throw new Error("Không tải được ảnh mẫu.");
      const sampleBlob = await sampleResponse.blob();
      result = await analyzeImage(
        new File([sampleBlob], "sample.jpg", {
          type: sampleBlob.type || "image/jpeg",
        }),
      );
    }
    showResult(result);
  } catch (e) {
    $("status").textContent = e.message; // báo lỗi, không bịa kết quả
    $("go").disabled = false;
    $("go").textContent = "Thử lại";
  }
};

// Drag and Drop Zone Handling
const drop = $("drop");
["dragenter", "dragover"].forEach((t) =>
  drop.addEventListener(t, (e) => {
    e.preventDefault();
    drop.classList.add("over");
  }),
);
["dragleave", "drop"].forEach((t) =>
  drop.addEventListener(t, (e) => {
    e.preventDefault();
    drop.classList.remove("over");
  }),
);
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  if (
    e.dataTransfer &&
    e.dataTransfer.files &&
    e.dataTransfer.files.length > 0
  ) {
    setImage(e.dataTransfer.files[0]);
  }
});
drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    $("file").click();
  }
});

// Dynamic Click Delegate for #again button
document.addEventListener("click", (e) => {
  if (e.target.id === "again") reset();
});

// Grad-CAM Controls
$("showOv").onchange = applyOv;
$("opa").oninput = () => {
  applyOv();
  const opaVal = $("opaVal");
  if (opaVal) opaVal.textContent = $("opa").value + "%";
};
