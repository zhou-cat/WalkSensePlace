const frame = document.getElementById("heatFrame");
const errorBox = document.getElementById("errorBox");
const openHeatBtn = document.getElementById("openHeatBtn");

openHeatBtn.addEventListener("click", () => {
  window.open(frame.src, "_blank", "noopener,noreferrer");
});

// Show/hide a helpful error overlay if the iframe fails
let loaded = false;

frame.addEventListener("load", () => {
  loaded = true;
  errorBox.classList.add("hidden");
});

frame.addEventListener("error", () => {
  errorBox.classList.remove("hidden");
});

// Fallback: if load never fires (missing file / blocked), show message
setTimeout(() => {
  if (!loaded) errorBox.classList.remove("hidden");
}, 1500);
