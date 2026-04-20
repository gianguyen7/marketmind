#!/usr/bin/env python3
"""Serve the MarketMind dashboard locally.

Usage (from anywhere):
    python src/dashboard/serve.py          # default port 8000
    python src/dashboard/serve.py 3000     # custom port

Then open the printed URL in your browser.
"""
import http.server
import os
import sys
import webbrowser

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

# Serve from project root so relative paths to outputs/ resolve correctly
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(root)

url = f"http://localhost:{PORT}/src/dashboard/"
print(f"MarketMind dashboard:  {url}")
print(f"Findings one-pager:    {url}one-pager.html")
print("Press Ctrl+C to stop.\n")

try:
    webbrowser.open(url)
except Exception:
    pass

handler = http.server.SimpleHTTPRequestHandler
httpd = http.server.HTTPServer(("", PORT), handler)

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\nStopped.")
    httpd.server_close()
