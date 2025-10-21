#!/usr/bin/env bash

if docker ps --format '{{.Names}}' | grep -q '^langflow-dev$'; then
  echo "LangFlow container is running."
else
  echo "LangFlow container is not running."
fi
