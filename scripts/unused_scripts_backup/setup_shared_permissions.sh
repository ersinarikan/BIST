#!/bin/bash
# Setup Shared Permissions for BIST Pattern
# Bu script shared group oluşturur ve dizinleri group-writable yapar

set -euo pipefail

ROOT_DIR="${1:-/opt/bist-pattern}"
SHARED_GROUP="bist-pattern"
WEB_USER="www-data"
WEB_GROUP="www-data"

echo "🔧 Setting up shared permissions for BIST Pattern..."

# 1. Create shared group if it doesn't exist
if ! getent group "$SHARED_GROUP" >/dev/null 2>&1; then
    echo "📦 Creating shared group: $SHARED_GROUP"
    groupadd -r "$SHARED_GROUP" || echo "⚠️ Group may already exist"
else
    echo "✅ Shared group already exists: $SHARED_GROUP"
fi

# 2. Add users to shared group
echo "👥 Adding users to shared group..."
usermod -a -G "$SHARED_GROUP" "$WEB_USER" 2>/dev/null || echo "⚠️ Could not add $WEB_USER to group (may already be member)"
usermod -a -G "$SHARED_GROUP" root 2>/dev/null || echo "⚠️ Could not add root to group (may already be member)"

# 3. Set up directories with group-writable permissions
echo "📁 Setting up directories..."

# Main directories
DIRS=(
    "$ROOT_DIR/results"
    "$ROOT_DIR/logs"
    "$ROOT_DIR/.cache"
    "$ROOT_DIR/.cache/enhanced_ml_models"
    "$ROOT_DIR/.cache/basic_ml_models"
    "$ROOT_DIR/results/continuous_hpo"
    "$ROOT_DIR/results/optuna_studies"
)

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "  📂 Creating directory: $dir"
        mkdir -p "$dir"
    fi
    
    # Set ownership to www-data:shared-group (or www-data:www-data if group doesn't exist)
    if getent group "$SHARED_GROUP" >/dev/null 2>&1; then
        chown -R "$WEB_USER:$SHARED_GROUP" "$dir" 2>/dev/null || chown -R "$WEB_USER:$WEB_GROUP" "$dir" || true
    else
        chown -R "$WEB_USER:$WEB_GROUP" "$dir" || true
    fi
    
    # Set permissions: 775 (rwxrwxr-x) for directories
    chmod -R 775 "$dir" 2>/dev/null || true
    
    echo "  ✅ $dir: $(stat -c '%U:%G %a' "$dir" 2>/dev/null || echo 'permissions set')"
done

# 4. Fix existing files in results and logs
echo "🔧 Fixing existing files..."

# Fix results directory files
if [ -d "$ROOT_DIR/results" ]; then
    find "$ROOT_DIR/results" -type f -user root -exec chown "$WEB_USER:$SHARED_GROUP" {} \; 2>/dev/null || \
    find "$ROOT_DIR/results" -type f -user root -exec chown "$WEB_USER:$WEB_GROUP" {} \; 2>/dev/null || true
    find "$ROOT_DIR/results" -type f -exec chmod 664 {} \; 2>/dev/null || true
    find "$ROOT_DIR/results" -type d -exec chmod 775 {} \; 2>/dev/null || true
fi

# Fix logs directory files
if [ -d "$ROOT_DIR/logs" ]; then
    find "$ROOT_DIR/logs" -type f -user root -exec chown "$WEB_USER:$SHARED_GROUP" {} \; 2>/dev/null || \
    find "$ROOT_DIR/logs" -type f -user root -exec chown "$WEB_USER:$WEB_GROUP" {} \; 2>/dev/null || true
    find "$ROOT_DIR/logs" -type f -exec chmod 664 {} \; 2>/dev/null || true
    find "$ROOT_DIR/logs" -type d -exec chmod 775 {} \; 2>/dev/null || true
fi

# Fix .cache directory
if [ -d "$ROOT_DIR/.cache" ]; then
    find "$ROOT_DIR/.cache" -type f -user root -exec chown "$WEB_USER:$SHARED_GROUP" {} \; 2>/dev/null || \
    find "$ROOT_DIR/.cache" -type f -user root -exec chown "$WEB_USER:$WEB_GROUP" {} \; 2>/dev/null || true
    find "$ROOT_DIR/.cache" -type f -exec chmod 664 {} \; 2>/dev/null || true
    find "$ROOT_DIR/.cache" -type d -exec chmod 775 {} \; 2>/dev/null || true
fi

echo ""
echo "✅ Shared permissions setup complete!"
echo ""
echo "📋 Summary:"
echo "  - Shared group: $SHARED_GROUP"
echo "  - Group members: $WEB_USER, root"
echo "  - Directory permissions: 775 (rwxrwxr-x)"
echo "  - File permissions: 664 (rw-rw-r--)"
echo ""
echo "💡 Note: New files created by root will be owned by root, but directories are group-writable."
echo "   Use the Python utility (bist_pattern.utils.file_permissions) to fix permissions automatically."

