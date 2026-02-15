// Prevent browser-level zoom so wheel gestures affect only Plotly charts.
// Chart zoom remains available via dcc.Graph config (scrollZoom=true).
(function () {
  function isEditableTarget(target) {
    if (!target) {
      return false;
    }
    var tag = target.tagName ? target.tagName.toUpperCase() : "";
    return tag === "INPUT" || tag === "TEXTAREA" || !!target.isContentEditable;
  }

  window.addEventListener(
    "wheel",
    function (event) {
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
      }
    },
    { passive: false, capture: true }
  );

  window.addEventListener(
    "keydown",
    function (event) {
      if (!(event.ctrlKey || event.metaKey)) {
        return;
      }
      if (isEditableTarget(event.target)) {
        return;
      }

      var key = event.key;
      if (key === "+" || key === "-" || key === "=" || key === "0") {
        event.preventDefault();
      }
    },
    { capture: true }
  );
})();
