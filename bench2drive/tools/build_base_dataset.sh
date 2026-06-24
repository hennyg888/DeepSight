#!/bin/bash
# Build training dataset from Bench2Drive-base (1001 tar.gz, ~312GB compressed).
# Stream-extracts every tarball into Bench2Drive-base-extracted/ and DELETES the
# tar.gz right after, then runs crop_bev + targetpointgen to produce
# /home/s56cai/DeepSight/data/train_bev_base.jsonl
#
# Disk math (measured): inflation ~1.38x -> extracted ~431GB. Net delta after
# deleting tarballs: +118GB. Should be safe on /home (~913GB free).
#
# Usage:
#   bash build_base_dataset.sh              # phase=all
#   bash build_base_dataset.sh extract      # phase=extract only
#   bash build_base_dataset.sh gunzip       # phase=gunzip anno/*.json.gz -> .json
#   bash build_base_dataset.sh crop         # phase=crop_bev only
#   bash build_base_dataset.sh jsonl        # phase=targetpointgen only

set -uo pipefail

TARBALL_DIR=/home/s56cai/DeepSight/bench2drive/Bench2Drive-base
EXTRACT_DIR=/home/s56cai/DeepSight/bench2drive/Bench2Drive-base-extracted
LOG=/home/s56cai/DeepSight/bench2drive/Bench2Drive-base-build.log
CONCURRENT=8                # parallel extract workers; bump if disk IO has headroom
MIN_FREE_GB=80              # pause extraction if free space drops below this
PHASE=${1:-all}             # all | extract | crop | jsonl

REPO=/home/s56cai/DeepSight
CROP_PY=$REPO/src/tools/crop_bev_for_bench2drive.py
TPG_PY=$REPO/bench2drive/dataprocess/targetpointgen.py

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

mkdir -p "$EXTRACT_DIR"

###############################################################################
# Phase 1: stream-extract tarballs
###############################################################################
phase_extract() {
  mapfile -t files < <(ls -1 "$TARBALL_DIR"/*.tar.gz 2>/dev/null)
  local total=${#files[@]}
  if (( total == 0 )); then
    log "no tarballs found in $TARBALL_DIR (already extracted?)"
    return 0
  fi
  log "Phase 1: extracting $total tarballs from $TARBALL_DIR (concurrency=$CONCURRENT)"

  extract_one() {
    local f=$1 idx=$2 total=$3
    local name
    name=$(basename "$f")
    if tar -xzf "$f" -C "$EXTRACT_DIR" 2>>"$LOG"; then
      rm -f "$f"
      printf '[%s] [%4d/%d] OK  %s\n' "$(ts)" "$idx" "$total" "$name" >>"$LOG"
    else
      printf '[%s] [%4d/%d] FAIL %s\n' "$(ts)" "$idx" "$total" "$name" >>"$LOG"
    fi
  }

  local i started=0
  for ((i=0; i<total; i++)); do
    extract_one "${files[$i]}" "$((i+1))" "$total" &
    started=$((started+1))
    if (( started % CONCURRENT == 0 )); then
      wait
      # disk safety check after every batch
      local free_gb
      free_gb=$(df -BG /home | awk 'NR==2 {gsub("G","",$4); print $4}')
      if (( free_gb < MIN_FREE_GB )); then
        log "WARN: free=${free_gb}GB < ${MIN_FREE_GB}GB threshold, pausing 60s"
        sleep 60
      fi
    fi
  done
  wait
  log "Phase 1 done. scenes extracted: $(ls -1 "$EXTRACT_DIR" 2>/dev/null | wc -l)"
}

###############################################################################
# Phase 1.5: gunzip anno/*.json.gz -> anno/*.json
# targetpointgen.py only reads files matching '*.json' suffix; base ships
# them as '.json.gz', so we decompress in-place (keep originals).
###############################################################################
phase_gunzip() {
  log "Phase 1.5: decompressing anno/*.json.gz under $EXTRACT_DIR"
  local before
  before=$(find "$EXTRACT_DIR" -name '*.json.gz' 2>/dev/null | wc -l)
  log "  found $before .json.gz files"
  if (( before == 0 )); then
    log "  nothing to do"
    return 0
  fi
  # 64-way parallel, 100 files per gunzip invocation; keep .gz originals
  find "$EXTRACT_DIR" -name '*.json.gz' -print0 \
    | xargs -0 -P 64 -n 100 gunzip -k 2>>"$LOG"
  local after
  after=$(find "$EXTRACT_DIR" -name '*.json' 2>/dev/null | wc -l)
  log "Phase 1.5 done. .json files now present: $after"
}

###############################################################################
# Phase 2: crop BEV (creates rgb_bev/ inside each scene)
###############################################################################
phase_crop() {
  log "Phase 2: crop_bev_for_bench2drive.py"
  cd "$REPO"
  python "$CROP_PY" 2>&1 | tee -a "$LOG"
  log "Phase 2 done"
}

###############################################################################
# Phase 3: targetpointgen -> data/train_bev_base.jsonl
###############################################################################
phase_jsonl() {
  log "Phase 3: targetpointgen.py"
  cd "$REPO/bench2drive/dataprocess"
  python "$TPG_PY" 2>&1 | tee -a "$LOG"
  local out=$REPO/data/train_bev_base.jsonl
  if [[ -f $out ]]; then
    local n
    n=$(wc -l <"$out")
    log "Phase 3 done. $out has $n lines"
  else
    log "ERROR: $out not produced"
    exit 1
  fi
}

###############################################################################
# dispatch
###############################################################################
case "$PHASE" in
  extract) phase_extract ;;
  gunzip)  phase_gunzip ;;
  crop)    phase_crop ;;
  jsonl)   phase_jsonl ;;
  all)     phase_extract; phase_gunzip; phase_crop; phase_jsonl ;;
  *)       echo "unknown phase: $PHASE (expected: all|extract|gunzip|crop|jsonl)"; exit 2 ;;
esac

log "ALL DONE for phase=$PHASE"
