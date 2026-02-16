# Install all dependencies
uv sync

# Set VSCode's default interpreter path
cd ..
mkdir -p .vscode

echo '{"python.defaultInterpreterPath":"/home/onyxia/work/crospint/.venv/bin/python"}' >> .vscode/settings.json
