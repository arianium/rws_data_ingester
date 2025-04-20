#!/bin/bash

set -e

source ./venv/bin/activate

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

python src/rijnhaven_swimming_advice.py

git add index.html
git commit -m "Auto-update report: $(date '+%Y-%m-%d %H:%M')"
# Make sure you locally tell git what key to use
# git config core.sshCommand "ssh -i /path/to/ssh/key -o IdentitiesOnly=yes"
git push origin main
