document.addEventListener("DOMContentLoaded", () => {
  const pressureBtn = document.getElementById("pressuremapBtn");
  const heatmapBtn = document.getElementById("heatmapBtn");
  const dataBtn = document.getElementById("dataBtn");
  //const footanimeBtn = document.getElementById("footanimeBtn");

  if (pressureBtn) {
    pressureBtn.addEventListener("click", () => {
      window.location.href = "pages/pressure.html";
    });
  }

  if (heatmapBtn) {
    heatmapBtn.addEventListener("click", () => {
      window.location.href = "pages/heat.html";
    });
  }

  if (dataBtn) {
    dataBtn.addEventListener("click", () => {
      window.location.href = "pages/**";
    });
  }

    /* temperarly commented out
   if (footanimeBtn) {
    footanimeBtn.addEventListener("click", () => {
      window.location.href = "pages/foot.html";
    });
  }
  */ 
});