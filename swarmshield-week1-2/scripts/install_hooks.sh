#!/bin/sh
set -e
git config core.hooksPath .githooks
chmod +x .githooks/*
echo "Git hooks enabled (commit-msg policy + pre-push checks)."
