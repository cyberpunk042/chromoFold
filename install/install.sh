#!/bin/sh
set -eu

PREFIX=${CHROMOFOLD_HOME:-"$HOME/.local/share/chromofold"}
BIN_DIR=${CHROMOFOLD_BIN:-"$HOME/.local/bin"}
SOURCE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

mkdir -p "$PREFIX" "$BIN_DIR"
for path in tools hub product docs; do
  rm -rf "$PREFIX/$path"
  cp -R "$SOURCE_DIR/$path" "$PREFIX/$path"
done

cat > "$BIN_DIR/chromofold" <<EOF
#!/bin/sh
exec python3 "$PREFIX/tools/chromofold.py" "\$@"
EOF
cat > "$BIN_DIR/chromofold-assistant" <<EOF
#!/bin/sh
exec python3 "$PREFIX/tools/chromofold_assistant.py" "\$@"
EOF
cat > "$BIN_DIR/chromofold-hub" <<EOF
#!/bin/sh
exec python3 "$PREFIX/hub/server.py" "\$@"
EOF
chmod 755 "$BIN_DIR/chromofold" "$BIN_DIR/chromofold-assistant" "$BIN_DIR/chromofold-hub"
printf '%s\n' "Installed ChromoFold in $PREFIX" "Commands: chromofold, chromofold-assistant, chromofold-hub" "Add $BIN_DIR to PATH when necessary."
