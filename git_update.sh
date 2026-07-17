#!/bin/zsh
set -e
git pull origin main --allow-unrelated-histories
git add .
git commit -m "$1"
git push origin main
