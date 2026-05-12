import csv
import hashlib
import json
import os
import smtplib
import ssl
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import yaml


ROOT = Path(".")
MODELS_FILE = ROOT / "models.yaml"
STATE_FILE = ROOT / "logs" / "alerts" / "decision_state.json"

GITHUB_TOKEN = os.getenv("GITHUB_READ_TOKEN", "").strip()

SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "").strip()
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "").strip()

SEND_ON_INITIAL_BASELINE = os.getenv("SEND_ON_INITIAL_BASELINE", "false").lower() == "true"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_text(repo: str, branch: str, path: str) -> str:
    url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path.lstrip('/')}"
    req = urllib.request.Request(url)

    if GITHUB_TOKEN:
        req.add_header("Authorization", f"Bearer {GITHUB_TOKEN}")

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def parse_csv_last_row(text: str) -> dict:
    if not text.strip():
        return {}

    rows = list(csv.DictReader(text.splitlines()))
    return rows[-1] if rows else {}


def stable_hash(*parts: str) -> str:
    joined = "\n---PART---\n".join(part or "" for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def load_models() -> list[dict]:
    with open(MODELS_FILE, "r") as f:
        data = yaml.safe_load(f)

    return [
        model for model in data.get("models", [])
        if model.get("enabled", True)
    ]


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}

    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))


def send_email(subject: str, body: str) -> None:
    if not all([SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, ALERT_EMAIL_FROM, ALERT_EMAIL_TO]):
        print("Email secrets are missing. Alert not sent.")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = ALERT_EMAIL_FROM
    msg["To"] = ALERT_EMAIL_TO
    msg.set_content(body)

    context = ssl.create_default_context()

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls(context=context)
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)


def build_model_snapshot(model: dict) -> dict:
    repo = model["repo"]
    branch = model.get("branch", "main")
    logs_path = model.get("logs_path", "logs").rstrip("/")

    latest_decision_text = fetch_text(repo, branch, f"{logs_path}/decisions/latest_decision.csv")
    decisions_history_text = fetch_text(repo, branch, f"{logs_path}/decisions/decisions.csv")
    latest_target_weights_text = fetch_text(repo, branch, f"{logs_path}/target_weights/latest_target_weights.csv")

    latest_decision = parse_csv_last_row(latest_decision_text)

    fingerprint = stable_hash(
        latest_decision_text,
        decisions_history_text,
        latest_target_weights_text,
    )

    return {
        "id": model.get("id"),
        "name": model.get("name") or model.get("id"),
        "repo": repo,
        "branch": branch,
        "logs_path": logs_path,
        "fingerprint": fingerprint,
        "latest_decision": latest_decision,
        "decision_text_present": bool(latest_decision_text.strip()),
        "target_weights_present": bool(latest_target_weights_text.strip()),
        "checked_at": now_utc(),
    }


def format_alert(snapshot: dict, previous: dict | None) -> tuple[str, str]:
    decision = snapshot.get("latest_decision", {}) or {}

    subject = f"QSentia Decision Change: {snapshot['name']}"

    body = f"""
QSentia detected a decision or strategy-state change.

Model: {snapshot['name']}
Model ID: {snapshot['id']}
Repo: {snapshot['repo']}
Checked At: {snapshot['checked_at']}

Latest Decision:
- Market Date: {decision.get('market_date', 'N/A')}
- Variant: {decision.get('variant', 'N/A')}
- Action: {decision.get('action', 'N/A')}
- Action Index: {decision.get('action_idx', 'N/A')}
- Last Action: {decision.get('last_action', 'N/A')}
- Submit Orders: {decision.get('submit_orders', 'N/A')}
- Account Status: {decision.get('account_status', 'N/A')}
- Account Value: {decision.get('account_value', 'N/A')}
- Target Positions: {decision.get('n_target_positions', 'N/A')}
- Orders Planned: {decision.get('n_orders_planned', 'N/A')}
- Orders Submitted: {decision.get('n_orders_submitted', 'N/A')}
- Timestamp UTC: {decision.get('timestamp_utc', 'N/A')}

Change Detection:
- Previous fingerprint: {(previous or {}).get('fingerprint', 'None')}
- New fingerprint: {snapshot.get('fingerprint')}
- Latest decision file present: {snapshot.get('decision_text_present')}
- Latest target weights file present: {snapshot.get('target_weights_present')}

This alert fires when latest_decision.csv, decisions.csv, or latest_target_weights.csv changes.
""".strip()

    return subject, body


def main() -> None:
    models = load_models()
    state = load_state()

    alerts_sent = 0
    new_state = dict(state)

    for model in models:
        model_id = model["id"]
        snapshot = build_model_snapshot(model)

        if not snapshot["decision_text_present"]:
            print(f"Skipping {model_id}: no latest decision file found.")
            continue

        previous = state.get(model_id)
        previous_fingerprint = (previous or {}).get("fingerprint")
        new_fingerprint = snapshot["fingerprint"]

        is_initial = previous_fingerprint is None
        changed = previous_fingerprint != new_fingerprint

        if changed and (not is_initial or SEND_ON_INITIAL_BASELINE):
            subject, body = format_alert(snapshot, previous)
            send_email(subject, body)
            alerts_sent += 1
            print(f"Alert sent for {model_id}")
        else:
            print(f"No change for {model_id}")

        new_state[model_id] = snapshot

    save_state(new_state)
    print(f"Decision alert check complete. Alerts sent: {alerts_sent}")


if __name__ == "__main__":
    main()
