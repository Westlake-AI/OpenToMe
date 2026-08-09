#!/usr/bin/env bash
# fetch_papers.sh — 从 manifest.tsv 重建 papers/ 目录
# 用法:
#   bash fetch_papers.sh            # 只下载缺失的 PDF
#   bash fetch_papers.sh --force    # 重新下载全部（覆盖）
#   bash fetch_papers.sh --check    # 只校验现有 PDF，不下载
# 依赖: curl, python3(可选，用于 PDF 校验)。无需联网即可 --check。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$HERE/manifest.tsv"
DEST="$HERE/papers"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"
MODE="${1:-}"

mkdir -p "$DEST"
[ -f "$MANIFEST" ] || { echo "ERROR: manifest not found: $MANIFEST"; exit 1; }

is_pdf() { [ -s "$1" ] && [ "$(file -b "$1" 2>/dev/null | cut -c1-3)" = "PDF" ]; }

ok=0; skip=0; fail=0; manual=0
while IFS=$'\t' read -r fn type src group title; do
  # skip comments / blank lines
  [[ "$fn" =~ ^#.*$ || -z "${fn// }" ]] && continue
  out="$DEST/$fn"

  if [ "$MODE" = "--check" ]; then
    if is_pdf "$out"; then echo "OK    $fn"; ok=$((ok+1)); else echo "MISS  $fn"; fail=$((fail+1)); fi
    continue
  fi

  if [ "$type" = "manual" ]; then
    if is_pdf "$out"; then echo "OK    $fn (manual, present)"; ok=$((ok+1));
    else echo "MANUAL  $fn -> 浏览器下载: $src"; manual=$((manual+1)); fi
    continue
  fi

  if [ "$MODE" != "--force" ] && is_pdf "$out"; then
    echo "SKIP  $fn (already valid)"; skip=$((skip+1)); continue
  fi

  case "$type" in
    arxiv) url="https://arxiv.org/pdf/$src" ;;
    url)   url="$src" ;;
    *)     echo "WARN  unknown type '$type' for $fn"; fail=$((fail+1)); continue ;;
  esac

  echo -n "GET   $fn <- $url ... "
  curl -sL -A "$UA" --max-time 120 -o "$out" "$url"
  if is_pdf "$out"; then echo "OK ($(du -h "$out"|cut -f1))"; ok=$((ok+1));
  else echo "FAIL (not a PDF)"; rm -f "$out"; fail=$((fail+1)); fi
done < "$MANIFEST"

echo "-----------------------------------------------"
echo "done: ok=$ok skip=$skip manual=$manual fail=$fail"
[ "$manual" -gt 0 ] && echo "注: manual 条目需按上面链接用浏览器手动下载到 papers/（bioRxiv Cloudflare 拦截脚本）。"

# 可选: pypdf 深度校验（页数）
if command -v python3 >/dev/null 2>&1 && python3 -c "import pypdf" 2>/dev/null; then
  echo "--- pypdf 页数校验 ---"
  python3 - "$DEST" <<'PY'
import sys, glob, os, pypdf
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.pdf"))):
    try:
        n = len(pypdf.PdfReader(f).pages); print(f"  {n:>3d}p  {os.path.basename(f)}")
    except Exception as e:
        print(f"  ERR  {os.path.basename(f)}: {e}")
PY
fi
