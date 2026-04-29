"""
One-time helper to obtain a Meta access token for the Ad Library API.

The Ad Library API (/ads_archive) is a public endpoint — it does NOT require
special Marketing API permissions like ads_read.  A plain App Access Token
(APP_ID|APP_SECRET) is sufficient for most queries including POLITICAL_AND_ISSUE_ADS.

TWO MODES:

  --mode app   (default, easiest)
      Builds an App Access Token from your App ID + App Secret.
      No browser, no OAuth flow, works immediately.
      Use this first.

  --mode user
      Runs a local OAuth flow to get a User Access Token.
      Only needed if app-token results are missing impressions/spend.
      Requires http://localhost:8080/ in Valid OAuth Redirect URIs.

Prerequisites (both modes):
  - App ID and App Secret from: App Dashboard → Settings → Basic

Usage:
  python get_token.py --app-id YOUR_APP_ID --app-secret YOUR_APP_SECRET
  python get_token.py --app-id YOUR_APP_ID --app-secret YOUR_APP_SECRET --mode user

Alternatively set env vars:
  export META_APP_ID=YOUR_APP_ID
  export META_APP_SECRET=YOUR_APP_SECRET
  python get_token.py
"""

from __future__ import annotations

import argparse
import http.server
import os
import urllib.parse
import webbrowser
from pathlib import Path

import requests


PORT = 8080
REDIRECT_URI = f"http://localhost:{PORT}/"
# No special scope needed — Ad Library API is public
SCOPE = "public_profile"
AUTH_URL = "https://www.facebook.com/dialog/oauth"
TOKEN_URL = "https://graph.facebook.com/v21.0/oauth/access_token"
ENV_FILE = Path(".env")


# ---------------------------------------------------------------------------
# App Access Token (no OAuth needed)
# ---------------------------------------------------------------------------

def get_app_token(app_id: str, app_secret: str) -> str:
    """
    Fetch an App Access Token from Meta.
    This is equivalent to the string 'APP_ID|APP_SECRET' but validated by Meta.
    """
    resp = requests.get(
        TOKEN_URL,
        params={
            "client_id": app_id,
            "client_secret": app_secret,
            "grant_type": "client_credentials",
        },
        timeout=15,
    )
    data = resp.json()
    if "access_token" not in data:
        raise RuntimeError(f"App token request failed: {data}")
    return data["access_token"]


# ---------------------------------------------------------------------------
# Tiny one-shot HTTP server (user token path only)

_captured_code: str | None = None


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        global _captured_code
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if "code" in params:
            _captured_code = params["code"][0]
            body = b"<h2>Authorization successful! You can close this tab.</h2>"
            self.send_response(200)
        elif "error" in params:
            err = params.get("error_description", ["unknown error"])[0]
            body = f"<h2>Authorization failed: {err}</h2>".encode()
            self.send_response(400)
        else:
            body = b"<h2>Waiting...</h2>"
            self.send_response(200)

        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:  # silence request logs
        pass


# ---------------------------------------------------------------------------
# Token exchange
# ---------------------------------------------------------------------------

def exchange_code(app_id: str, app_secret: str, code: str) -> str:
    resp = requests.get(
        TOKEN_URL,
        params={
            "client_id": app_id,
            "client_secret": app_secret,
            "redirect_uri": REDIRECT_URI,
            "code": code,
        },
        timeout=15,
    )
    data = resp.json()
    if "access_token" not in data:
        raise RuntimeError(f"Token exchange failed: {data}")
    return data["access_token"]


def write_env(token: str) -> None:
    """Write or update META_ACCESS_TOKEN in .env."""
    lines: list[str] = []
    found = False

    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("META_ACCESS_TOKEN="):
                lines.append(f"META_ACCESS_TOKEN={token}")
                found = True
            else:
                lines.append(line)

    if not found:
        lines.append(f"META_ACCESS_TOKEN={token}")

    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✓ Token written to {ENV_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Obtain a Meta access token.")
    parser.add_argument(
        "--app-id",
        default=os.environ.get("META_APP_ID", ""),
        help="Your Meta App ID (App Dashboard → Settings → Basic)",
    )
    parser.add_argument(
        "--app-secret",
        default=os.environ.get("META_APP_SECRET", ""),
        help="Your Meta App Secret (App Dashboard → Settings → Basic)",
    )
    parser.add_argument(
        "--mode",
        choices=["app", "user"],
        default="app",
        help="'app' = App Access Token (default, no browser needed); 'user' = User OAuth flow",
    )
    args = parser.parse_args()

    if not args.app_id:
        args.app_id = input("Enter your App ID: ").strip()
    if not args.app_secret:
        args.app_secret = input("Enter your App Secret: ").strip()

    if not args.app_id or not args.app_secret:
        print("Error: App ID and App Secret are required.")
        return

    if args.mode == "app":
        print("Fetching App Access Token (no browser needed)...")
        token = get_app_token(args.app_id, args.app_secret)
        print(f"Token obtained (first 20 chars): {token[:20]}...")
        write_env(token)
        print("\nAll done. Run the dry-run to confirm it works:")
        print("  python run_scrape.py --dry-run")
        return

    # --- user token path ---
    auth_params = urllib.parse.urlencode({
        "client_id": args.app_id,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "response_type": "code",
    })
    full_url = f"{AUTH_URL}?{auth_params}"

    print("\n=== Step 1 ===")
    print(f"Make sure  http://localhost:{PORT}/  is in your app's")
    print("Valid OAuth Redirect URIs:")
    print("  App Dashboard → Facebook Login → Settings → Valid OAuth Redirect URIs")
    print()
    print("=== Step 2 ===")
    print("Opening browser for Facebook authorization...")
    webbrowser.open(full_url)
    print(f"(If the browser didn't open, go to:\n  {full_url})\n")

    print(f"=== Step 3 — Waiting for callback on http://localhost:{PORT}/ ===")
    server = http.server.HTTPServer(("localhost", PORT), _Handler)
    server.handle_request()

    if _captured_code is None:
        print("No authorization code received. Aborting.")
        return

    print("Authorization code received. Exchanging for access token...")
    token = exchange_code(args.app_id, args.app_secret, _captured_code)
    print(f"Token obtained (first 20 chars): {token[:20]}...")

    write_env(token)
    print("\nAll done. Run the dry-run to confirm it works:")
    print("  python run_scrape.py --dry-run")


if __name__ == "__main__":
    main()
