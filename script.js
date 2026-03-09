document.addEventListener("DOMContentLoaded", () => {
  const pressureBtn = document.getElementById("pressuremapBtn");
  const heatmapBtn = document.getElementById("heatmapBtn");
  const footanimeBtn = document.getElementById("footanimeBtn");

  if (pressureBtn) {
    pressureBtn.addEventListener("click", () => {
      window.location.href = "../pages - visuals/pressure.html";
    });
  }

  if (heatmapBtn) {
    heatmapBtn.addEventListener("click", () => {
      window.location.href = "../pages - visuals/heat.html";
    });
  }

  if (footanimeBtn) {
    footanimeBtn.addEventListener("click", () => {
      window.location.href = "../pages - visuals/foot.html";
    });
  }
});