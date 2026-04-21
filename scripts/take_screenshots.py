"""Capture screenshots of the dashboard and one-pager for README.

Requires: pip install playwright && playwright install chromium
Usage: python scripts/take_screenshots.py
"""
import asyncio
from pathlib import Path

async def main():
    from playwright.async_api import async_playwright

    out = Path("docs/screenshots")
    out.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1200, "height": 900})

        # One-pager (full page, letter-sized)
        await page.goto("http://localhost:8000/src/dashboard/one-pager.html")
        await page.wait_for_timeout(2000)
        element = page.locator(".page")
        await element.screenshot(path=str(out / "one-pager.png"))
        print(f"Saved: {out / 'one-pager.png'}")

        # Dashboard overview
        await page.goto("http://localhost:8000/src/dashboard/index.html")
        await page.wait_for_timeout(3000)
        await page.screenshot(path=str(out / "dashboard.png"))
        print(f"Saved: {out / 'dashboard.png'}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
