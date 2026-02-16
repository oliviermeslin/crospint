# Install all dependencies
uv sync

# Set VSCode's default interpreter path
cd ..
mkdir -p .vscode

echo '{"python.defaultInterpreterPath":"/home/onyxia/work/crospint/.venv/bin/python", "python.terminal.activateEnvironment": true}' >> .vscode/settings.json
