#!/usr/bin/env python3
"""Re-enumerate only the RARS01 STM32 USB CDC device.

This deliberately resets the STM32 (VID:PID 0483:5740), not the whole Jetson
Type-C controller, so an RGB-D camera on another USB path is not disturbed.
Motors must already be disabled before this script is executed.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


USB_DEVICES = Path("/sys/bus/usb/devices")
STM32_VENDOR_ID = "0483"
STM32_PRODUCT_ID = "5740"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii").strip().lower()
    except OSError:
        return ""


def _find_stm32() -> list[Path]:
    found: list[Path] = []
    for device in USB_DEVICES.iterdir():
        if (
            _read_text(device / "idVendor") == STM32_VENDOR_ID
            and _read_text(device / "idProduct") == STM32_PRODUCT_ID
            and (device / "authorized").is_file()
        ):
            found.append(device)
    return sorted(found)


def _tty_names(device: Path) -> list[str]:
    return sorted(path.name for path in device.glob("**/ttyACM*"))


def _wait_for_tty(device: Path, timeout_s: float) -> list[str]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        names = _tty_names(device)
        if names and all((Path("/dev") / name).exists() for name in names):
            return names
        time.sleep(0.1)
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset only the RARS01 STM32 USB CDC device")
    parser.add_argument("--execute", action="store_true", help="perform the USB reset")
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    args = parser.parse_args()

    devices = _find_stm32()
    if len(devices) != 1:
        print(f"Expected one STM32 {STM32_VENDOR_ID}:{STM32_PRODUCT_ID}; found {len(devices)}")
        return 1
    device = devices[0]
    names = _tty_names(device)
    print(f"STM32 USB device: {device.name}" + (f" ({', '.join(names)})" if names else ""))

    if not args.execute:
        print("Dry run. With motors disabled: sudo .venv/bin/python scripts/reset_stm32_usb.py --execute")
        return 0
    if os.geteuid() != 0:
        print("Run with sudo; writing USB authorized requires root")
        return 1
    if args.settle_s < 0.0 or args.timeout_s <= 0.0:
        print("settle-s must be non-negative and timeout-s must be positive")
        return 1

    authorized = device / "authorized"
    try:
        authorized.write_text("0\n", encoding="ascii")
        time.sleep(args.settle_s)
        authorized.write_text("1\n", encoding="ascii")
    except OSError as exc:
        print(f"USB reset failed: {exc}")
        return 1

    names = _wait_for_tty(device, args.timeout_s)
    if not names:
        print("STM32 was reset but no /dev/ttyACM* appeared")
        return 1
    print("STM32 reconnected: " + ", ".join(f"/dev/{name}" for name in names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
