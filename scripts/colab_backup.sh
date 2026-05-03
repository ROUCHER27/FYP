%%bash
set -euo pipefail

# === 紧急备份脚本：把当前运行记录存到 Drive ===
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/content/drive/MyDrive/FYP/phase2/backups/run_${TIMESTAMP}"
mkdir -p "$BACKUP_DIR"

echo "=========================================="
echo "Backing up run records to Drive"
echo "  Target: $BACKUP_DIR"
echo "=========================================="

# 1. 复制 Phase 2 实验结果（如果存在）
SRC="/content/drive/MyDrive/FYP/phase2/loss_component_analysis"
if [ -d "$SRC" ]; then
    cp -r "$SRC" "$BACKUP_DIR/loss_component_analysis"
    echo "✓ Copied loss_component_analysis"
else
    echo "✗ loss_component_analysis not found"
fi

# 2. 复制 Phase 2.2 历史结果（如果存在）
SRC22="/content/drive/MyDrive/FYP/phase2.2"
if [ -d "$SRC22" ]; then
    cp -r "$SRC22" "$BACKUP_DIR/phase2.2"
    echo "✓ Copied phase2.2"
else
    echo "✗ phase2.2 not found"
fi

# 3. 复制本地 sanity_outputs（可能有未同步的结果）
LOCAL="/content/FYP/sanity_outputs"
if [ -d "$LOCAL" ]; then
    cp -r "$LOCAL" "$BACKUP_DIR/sanity_outputs_local"
    echo "Copied local sanity_outputs"
fi

# 4. 复制 checkpoints（断点续训状态）
CKPTS="/content/FYP/sanity_outputs/checkpoints"
if [ -d "$CKPTS" ]; then
    cp -r "$CKPTS" "$BACKUP_DIR/checkpoints_local"
    echo "Copied local checkpoints"
fi

# 5. 保存 git 状态和 notebook 信息
cd /content/FYP
{
    echo "=== Git Status ==="
    git log --oneline -5
    echo ""
    echo "=== Branch ==="
    git branch --show-current
    echo ""
    echo "=== Uncommitted Changes ==="
    git status --short
    echo ""
    echo "=== Disk Usage ==="
    du -sh "$BACKUP_DIR"/*
} > "$BACKUP_DIR/backup_info.txt"

echo ""
echo "=========================================="
echo "Backup complete!"
echo "Location: $BACKUP_DIR"
echo "=========================================="
ls -lh "$BACKUP_DIR/"
