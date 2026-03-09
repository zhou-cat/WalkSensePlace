document.addEventListener("DOMContentLoaded", () => {

    const pressureBtn = document.getElementById("pressuremapBtn");
    const heatmapBtn = document.getElementById("heatmapBtn");
    const footanimeBtn = document.getElementById("footanimeBtn");
  
    if (pressureBtn) {
      pressureBtn.addEventListener("click", () => {
        window.open("../pages/pressure.html", "_blank");
      });
    }
  
    if (heatmapBtn) {
      heatmapBtn.addEventListener("click", () => {
        window.open("../pages/heat.html", "_blank");
      });
    }
  
    if (footanimeBtn) {
      footanimeBtn.addEventListener("click", () => {
        window.open("../pages/foot.html", "_blank");
      });
    }
  
  });
  