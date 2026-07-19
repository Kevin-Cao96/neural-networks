#!/bin/zsh
set -e
cd "$(dirname "$0")"
git pull origin main
git add .
git commit -m "${1:-update}"    # 传参就用参数，没传就用 "update"
git push origin main