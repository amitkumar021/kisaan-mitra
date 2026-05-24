// frontend/script.js

const API_URL = "http://localhost:5000/predict";

let selectedFile = null;

// Handle file selection
document.getElementById("file-input").addEventListener("change", function () {
  if (this.files[0]) handleFile(this.files[0]);
});

// Allow drag and drop on upload area
const uploadArea = document.getElementById("upload-area");
uploadArea.addEventListener("dragover", e => { e.preventDefault(); uploadArea.style.borderColor = "#1D9E75"; });
uploadArea.addEventListener("dragleave",  () => uploadArea.style.borderColor = "#ccc");
uploadArea.addEventListener("drop", e => {
  e.preventDefault();
  uploadArea.style.borderColor = "#ccc";
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
uploadArea.addEventListener("click", () => document.getElementById("file-input").click());

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById("preview-img").src = e.target.result;
    document.getElementById("upload-area").style.display    = "none";
    document.getElementById("preview-section").style.display = "block";
    document.getElementById("result-card").style.display    = "none";
    document.getElementById("error-box").style.display      = "none";
  };
  reader.readAsDataURL(file);
}

async function analyzeImage() {
  if (!selectedFile) return;

  const btn = document.getElementById("analyze-btn");
  btn.disabled   = true;
  btn.textContent = "Analyzing...";

  document.getElementById("loading").style.display       = "flex";
  document.getElementById("preview-section").style.display = "none";
  document.getElementById("error-box").style.display     = "none";
  document.getElementById("result-card").style.display   = "none";

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch(API_URL, { method: "POST", body: formData });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || "Server error");
    }

    const data = await response.json();
    showResult(data);

  } catch (err) {
    document.getElementById("error-box").style.display = "block";
    document.getElementById("error-box").textContent =
      "Error: " + err.message + ". Make sure the Flask server is running on port 5000.";
    document.getElementById("preview-section").style.display = "block";
  } finally {
    document.getElementById("loading").style.display = "none";
    btn.disabled    = false;
    btn.textContent = "Analyze Plant";
  }
}

function showResult(data) {
  const isHealthy = data.is_healthy;

  document.getElementById("result-header").className =
    "result-header " + (isHealthy ? "healthy" : "diseased");

  document.getElementById("result-disease").textContent = data.disease;
  document.getElementById("result-plant").textContent   = "Plant: " + data.plant;

  const badge = document.getElementById("confidence-badge");
  badge.textContent = data.confidence.toFixed(1) + "%";
  badge.className   = "confidence-badge" + (data.confidence < 60 ? " low" : "");

  const statusTag = document.getElementById("status-tag");
  statusTag.textContent = isHealthy ? "Healthy" : "Disease detected";
  statusTag.className   = "status-tag " + (isHealthy ? "healthy" : "diseased");

  document.getElementById("result-treatment").textContent = data.treatment;

  // Top 3 predictions
  const top3Html = data.top3.map(item => {
    const parts  = item.class.split("___");
    const label  = (parts[1] || parts[0]).replace(/_/g, " ");
    const width  = item.confidence.toFixed(1);
    return `
      <div class="top3-item">
        <span class="top3-name">${label}</span>
        <div class="top3-bar-wrap"><div class="top3-bar" style="width:${width}%"></div></div>
        <span class="top3-pct">${width}%</span>
      </div>`;
  }).join("");
  document.getElementById("top3-list").innerHTML = top3Html;

  document.getElementById("result-card").style.display = "block";
}

function resetApp() {
  selectedFile = null;
  document.getElementById("file-input").value     = "";
  document.getElementById("upload-area").style.display    = "block";
  document.getElementById("preview-section").style.display = "none";
  document.getElementById("result-card").style.display    = "none";
  document.getElementById("error-box").style.display      = "none";
  document.getElementById("loading").style.display        = "none";
}