#!/bin/bash
# Serve the local dashboard preview: http://localhost:8765/preview.html (this PC only).
cd "$(dirname "$0")/../dashboard" && exec python -m http.server 8765 --bind 127.0.0.1
