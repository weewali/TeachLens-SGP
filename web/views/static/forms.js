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

// Planned activities (optional): each button adds a row with one box for how many minutes it lasted; Remove deletes it.
// Without scripts the buttons stay hidden and the form shows one plain row with a list to choose from instead.
(function () {
  const add = document.getElementById("activity-add");
  const rows = document.getElementById("activity-rows");
  const template = document.getElementById("activity-row");
  if (!add || !rows || !template) return;
  add.hidden = false;

  add.addEventListener("click", function (event) {
    const button = event.target.closest("button[data-activity]");
    if (!button) return;
    const row = template.content.firstElementChild.cloneNode(true);
    const name = button.textContent.replace(/^\+\s*/, "").trim();
    row.querySelector("input[name=activity_type]").value = button.dataset.activity;
    row.querySelector(".activity-name").textContent = name;
    row.querySelector(".activity-remove").setAttribute("aria-label", "Remove " + name.toLowerCase());
    const minutes = row.querySelector("input[name=activity_minutes]");
    minutes.setAttribute("aria-label", "Duration of the " + name.toLowerCase() + " in minutes");
    rows.appendChild(row);
    minutes.focus();
  });

  rows.addEventListener("click", function (event) {
    const remove = event.target.closest(".activity-remove");
    if (!remove) return;
    remove.closest(".activity-row").remove();
    const first = add.querySelector("button");
    if (first) first.focus();
  });
})();
