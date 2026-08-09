#!/usr/bin/env bash
# clone_repos.sh — 从 manifest.tsv 重建 repos/ 目录
# 用法:
#   bash clone_repos.sh            # 只克隆缺失的仓库
#   bash clone_repos.sh --force    # 删除已存在目录后重新克隆
#   bash clone_repos.sh --keep-git # 克隆后保留 .git（默认删除以省空间）
#   bash clone_repos.sh --check    # 只检查哪些仓库存在/缺失
# 依赖: git, curl。默认 --depth 1 浅克隆 + 跳过 Git-LFS 大文件。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$HERE/manifest.tsv"
DEST="$HERE/repos"
export GIT_LFS_SKIP_SMUDGE=1   # 跳过 LFS（大权重文件），只要代码

FORCE=0; KEEP_GIT=0; CHECK=0
for a in "$@"; do
  case "$a" in
    --force) FORCE=1 ;;
    --keep-git) KEEP_GIT=1 ;;
    --check) CHECK=1 ;;
  esac
done

mkdir -p "$DEST"
[ -f "$MANIFEST" ] || { echo "ERROR: manifest not found: $MANIFEST"; exit 1; }

ok=0; skip=0; fail=0
while IFS=$'\t' read -r name url role note; do
  [[ "$name" =~ ^#.*$ || -z "${name// }" ]] && continue
  target="$DEST/$name"

  if [ "$CHECK" = 1 ]; then
    if [ -d "$target" ]; then echo "OK    $name ($(du -sh "$target" 2>/dev/null|cut -f1))"; ok=$((ok+1));
    else echo "MISS  $name  <- $url"; fail=$((fail+1)); fi
    continue
  fi

  if [ -d "$target" ]; then
    if [ "$FORCE" = 1 ]; then echo "RM    $name (force)"; rm -rf "$target";
    else echo "SKIP  $name (exists)"; skip=$((skip+1)); continue; fi
  fi

  echo ">>> clone $name <- $url"
  if git clone --depth 1 "$url" "$target" 2>&1 | tail -1; then
    [ "$KEEP_GIT" = 0 ] && rm -rf "$target/.git"
    echo "    OK $name ($(du -sh "$target" 2>/dev/null|cut -f1))"; ok=$((ok+1))
  else
    echo "    FAIL $name"; fail=$((fail+1))
  fi
done < "$MANIFEST"

echo "-----------------------------------------------"
echo "done: ok=$ok skip=$skip fail=$fail"
echo "提示: 复现所需的关键子路径 —"
echo "  OpenToMe/trainer/flame/{configs,scripts/byte,flame/{train.py,data.py}}"
echo "  OpenToMe/opentome/models/{transformer,delta_net,mergenet_nlp}"
echo "  hyena-dna/src/dataloaders/datasets/hg38_dataset.py"
echo "  hnet/hnet/{models/hnet.py,modules/dc.py}"
echo "  DNABERT_2/finetune/  (GUE 微调)"
