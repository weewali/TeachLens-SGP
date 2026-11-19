// New-analysis form: show that the (possibly large) videos are uploading and stop a second click.
(function () {
  const form = document.getElementById("new-lecture");
  if (!form) return;
  form.addEventListener("submit", function () {
    const button = form.querySelector("button[type=submit]");
    if (button) { button.disabled = true; button.textContent = "Uploading…"; }
    const note = document.getElementById("uploading");
    if (note) note.hidden = false;
  });
})();
