"""
Zerodha Enctoken Extractor — Manual OTP Entry
===============================================
Logs into Zerodha Kite using your User ID, Password,
and the 6-digit OTP from your authenticator app.

No browser DevTools. No TOTP secret key needed.
Just enter the 6-digit code when prompted.

USAGE:
    python get_enctoken.py

INSTALL:
    pip install requests --break-system-packages
"""

import requests
import json
import getpass
import sys
from pathlib import Path


CONFIG_FILE = "config_ltcg.json"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":     "application/json, text/plain, */*",
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer":    "https://kite.zerodha.com/",
    "Origin":     "https://kite.zerodha.com",
})


def step1_login(user_id: str, password: str) -> str | None:
    """Send user ID + password. Returns request_id for 2FA step."""
    print("\n  [1/3] Sending credentials...", end=" ", flush=True)
    try:
        r = SESSION.post("https://kite.zerodha.com/api/login", data={
            "user_id":  user_id,
            "password": password,
        }, timeout=15)
        data = r.json()
        if data.get("status") == "success":
            print("✓")
            return data["data"]["request_id"]
        else:
            print(f"✗\n  Error: {data.get('message', 'Login failed')}")
            return None
    except Exception as e:
        print(f"✗\n  Connection error: {e}")
        return None


def step2_twofa(user_id: str, request_id: str, otp: str) -> bool:
    """Submit the 6-digit OTP for 2FA."""
    print("  [2/3] Submitting OTP...", end=" ", flush=True)
    try:
        r = SESSION.post("https://kite.zerodha.com/api/twofa", data={
            "user_id":      user_id,
            "request_id":   request_id,
            "twofa_value":  otp.strip(),
            "twofa_type":   "totp",
            "skip_session": "",
        }, timeout=15)
        data = r.json()
        if data.get("status") == "success":
            print("✓")
            return True
        else:
            print(f"✗\n  Error: {data.get('message', '2FA failed')}")
            return False
    except Exception as e:
        print(f"✗\n  Connection error: {e}")
        return False


def step3_get_enctoken() -> str | None:
    """Extract enctoken from session cookies after successful login."""
    print("  [3/3] Extracting enctoken...", end=" ", flush=True)

    # Primary: direct cookie lookup
    enctoken = SESSION.cookies.get("enctoken")
    if enctoken:
        print("✓")
        return enctoken

    # Fallback: scan all cookies
    for cookie in SESSION.cookies:
        if "enctoken" in cookie.name.lower():
            print("✓")
            return cookie.value

    print("✗\n  enctoken not found in cookies")
    return None


def save_to_config(enctoken: str):
    """Save enctoken into config.json using string replace — preserves all comments."""
    import re
    path = Path(CONFIG_FILE)

    # Always print full token so user can copy manually if needed
    print(f"\n{'─'*60}")
    print(f"  enctoken (full — copy if needed):")
    print(f"  {enctoken}")
    print(f"{'─'*60}\n")

    if not path.exists():
        print(f"  ⚠ {CONFIG_FILE} not found in current folder")
        _save_to_txt(enctoken)
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        # Replace whatever value "enctoken" currently holds
        # Matches:  "enctoken": "...anything..."
        updated = re.sub(
            r'("enctoken"\s*:\s*)"[^"]*"',
            rf'\g<1>"{enctoken}"',
            raw
        )

        if updated == raw:
            # enctoken line not found — append warning
            print("  ⚠ Could not find enctoken key in config.json")
            _save_to_txt(enctoken)
            return

        with open(path, "w", encoding="utf-8") as f:
            f.write(updated)

        print(f"  ✓ enctoken updated in {CONFIG_FILE} → zerodha.enctoken")

    except Exception as e:
        print(f"  ⚠ Error updating {CONFIG_FILE}: {e}")
        _save_to_txt(enctoken)


def _save_to_txt(enctoken: str):
    with open("enctoken.txt", "w") as f:
        f.write(enctoken)
    print("  ✓ Saved to enctoken.txt — paste manually into config.json")


def main():
    print("\n🔐 Zerodha Enctoken Extractor")
    print("   Enter your credentials below.")
    print("   Password and OTP input is hidden for security.\n")
    print("─" * 40)

    user_id  = input("  User ID          : ").strip().upper()
    #password = getpass.getpass("  Password (hidden): ")
    password  = input("  password       : ").strip()
    #password =''

    # Step 1 — send credentials
    request_id = step1_login(user_id, password)
    if not request_id:
        print("\n❌ Login failed. Check your User ID and password.")
        sys.exit(1)

    # Step 2 — get OTP from user
    print("\n  Open your Authenticator app (Google Auth / Zerodha Auth)")
    print("  and enter the current 6-digit OTP:\n")
    #otp = getpass.getpass("  6-digit OTP (hidden): ").strip()
    otp = input("  otp       : ").strip()

    if not otp.isdigit() or len(otp) != 6:
        print("❌ OTP must be exactly 6 digits.")
        sys.exit(1)

    success = step2_twofa(user_id, request_id, otp)
    if not success:
        print("\n❌ 2FA failed. OTP may have expired — run the script again and enter a fresh OTP.")
        sys.exit(1)

    # Step 3 — extract enctoken
    enctoken = step3_get_enctoken()
    if not enctoken:
        print("\n❌ Could not extract enctoken.")
        sys.exit(1)

    # Save it
    save_to_config(enctoken)

    print("✅ Done! You can now run:")
    print("   python zerodha_downloader.py\n")
    print("⚠  Note: enctoken expires when you logout or after ~8 hours.")
    print("   Re-run this script each trading session.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
