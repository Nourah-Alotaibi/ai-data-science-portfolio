# AI Web Security Scanner — extension prototype audit

**Yes: project 14 is a Chrome browser-extension prototype.** The original `manifest.json` declares Manifest V3, `activeTab` and `scripting`, with `popup.html` as the extension popup. Its JavaScript captures the current page's HTML and URL, then sends them to a local Flask endpoint. That endpoint calls modules named for GPT analysis, SecurityHeaders, OWASP ZAP and Shodan, and returns their reports and risk scores.

## What the files establish

- Extension UI: `manifest.json`, `popup.html`, `popup.js`.
- Backend: `app.py`, with five analysis/integration helper files.
- This is a source prototype. A working installed extension and end-to-end API execution have not been demonstrated in this session.

## Issues found in the collected source

1. `popup.js` is also registered as a background service worker, but uses `document.addEventListener`; extension service workers have no document. It belongs in the popup only unless a separate worker is implemented.
2. The manifest references `icon.png`, which is absent from the collected files.
3. Page capture and backend submission happen immediately when the popup opens. Add an explicit Analyze action, clear destination disclosure and controls for optional external services before real use.
4. The JavaScript does not handle inaccessible tabs, injection errors, missing results, connection failures or API errors.
5. The local-backend permission/access flow, credentials, timeouts and per-service errors need implementation and browser verification. Flask runs in debug mode in the prototype.
6. ZAP is invoked alongside passive checks. Active scanning must be an explicit action limited to systems the operator is authorized to test.

## Portfolio description

Designed a Chrome extension prototype that connects page inspection to a Python Flask security-analysis backend, combining planned AI-assisted explanations with security-header, ZAP and Shodan integrations. The project explores how findings from multiple tools can be presented in one interface. Current status: prototype requiring integration fixes and end-to-end validation.

**Skills:** JavaScript, Chrome Extensions, Python, Flask, API integration, web security.

Do not describe this as a production scanner or claim measured detection accuracy. No external scan was executed as part of this audit.
