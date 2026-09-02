#!/usr/bin/env bash

set -e

LIMIT=5242880

# Remove files from Git staging only. This never deletes local files.
git restore --staged . 2>/dev/null || true

# Keep generated Mac/IDE files out of Git.
for RULE in "Output/" "._*" ".idea/" ".ipynb_checkpoints/"; do
  grep -Fqx "$RULE" .gitignore || printf '%s\n' "$RULE" >> .gitignore
done

# Add every current file larger than 5 MB to .gitignore.
find . -type f -size +5242880c \
  ! -path "./.git/*" \
  ! -path "./Output/*" \
  -print0 |
while IFS= read -r -d '' FILE; do
  PATH_TO_IGNORE="${FILE#./}"
  grep -Fqx "$PATH_TO_IGNORE" .gitignore || printf '%s\n' "$PATH_TO_IGNORE" >> .gitignore
done

# Stage normal project files.
git add -A

# Safety rule: unstage any staged file larger than 5 MB.
while IFS= read -r -d '' FILE; do
  if [ -f "$FILE" ] && [ "$(stat -f%z "$FILE")" -gt "$LIMIT" ]; then
    git restore --staged -- "$FILE"
    echo "Kept local only because it is larger than 5 MB: $FILE"
  fi
done < <(git diff --cached --name-only -z)

echo
echo "Files ready for GitHub:"
git diff --cached --name-only

