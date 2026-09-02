#!/bin/bash
# DexteraAI healthcheck script
set -e
curl -f http://localhost:8000/api/health || exit 1
