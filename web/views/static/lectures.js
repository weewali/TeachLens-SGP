// Lecture list: a short-lived message (such as "… was deleted.") fades out after the time in its data-autohide
// attribute (milliseconds), and ?deleted=… is taken out of the address at once, so reloading the page does not show
// the message again. Without scripts the message simply stays until the next page.
(function () {
  "use strict";
  var FADE_MS = 400;   // the same length as the CSS transition
  var notices = document.querySelectorAll("[data-autohide]");
  if (!notices.length) return;

  try {
    var url = new URL(window.location.href);
    if (url.searchParams.has("deleted")) {
      url.searchParams.delete("deleted");
      history.replaceState(history.state, "", url.pathname + url.search + url.hash);
    }
  } catch (e) { /* an old browser: the message still fades */ }

  Array.prototype.forEach.call(notices, function (notice) {
    var showFor = parseInt(notice.getAttribute("data-autohide"), 10) || 5000;
    setTimeout(function () {
      notice.classList.add("is-fading");
      setTimeout(function () { notice.parentNode && notice.parentNode.removeChild(notice); }, FADE_MS);
    }, showFor);
  });
})();
