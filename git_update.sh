#!/bin/zsh
set -e
git pull origin main --allow-unrelated-histories
git add .
git commit -m "update"
git push origin main
