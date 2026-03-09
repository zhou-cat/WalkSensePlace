const plotSelect = document.getElementById("plotSelect");
const plotFrame = document.getElementById("plotFrame");
const plotTitle = document.getElementById("plotTitle");

function prettifyFilename(filename) {
  return filename
    .replace(".html", "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function setPlot(filename) {
  plotFrame.src = `plots/${filename}`;
  plotTitle.textContent = prettifyFilename(filename);
}

plotSelect.addEventListener("change", () => {
  setPlot(plotSelect.value);
});

setPlot(plotSelect.value);