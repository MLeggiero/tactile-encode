#!/usr/bin/env bash
# The friction study's Part 1, unattended and resumable (tactile_rl/README.md, "The friction study"): the hand's
# friction on the YCB hammer fixed per condition in the 8 kHz validator (residual_experiment stage Efric: mu 0.2-1.5,
# holds 1.0 and 0.5, 12 paired seeds), the zero residual and the students, with the residual's own activity.
#
#   CPU: sweep (zero, s250) -> report -> [s1k's ONNX] sweep (+ s1k) -> report
#
# then the final report (friction_study: report.md and the figures) and $OUT/STUDY_DONE. Every step leaves a marker; a
# rerun skips finished steps (and finished validator episodes). Part 2 (the retrained teacher and students, swept into
# the same results) is run_v3_retrain.sh. Settings: environment variables or $STUDY_ENV (VAR=value lines).
set -u

STUDY_ENV=${STUDY_ENV:-$HOME/residual40/friction.env}
# shellcheck disable=SC1090
[ -f "$STUDY_ENV" ] && . "$STUDY_ENV"

TREE=${TREE:-$HOME/residual40/tree_fric}
PIPE=${PIPE:-$HOME/residual40/runs}            # the main pipeline's outputs: library, grasps, students
OUT=${OUT:-$HOME/residual40/friction}
PY_RL=${PY_RL:-$HOME/tactile-encode/.venv-rl/bin/python}
PY_VAL=${PY_VAL:-$HOME/tactile-encode/.venv/bin/python}
WORKERS=${WORKERS:-10}
STAGE=${STAGE:-Efric}

LIB=$PIPE/lib/hammer_ycb_40d.npz
S250=$PIPE/export/student_s250.onnx
S1K=$PIPE/export/student_s1k.onnx
M=$OUT/markers
LOG=$OUT/study.log
mkdir -p "$OUT"/{logs,markers,e,reports}
export PYTHONPATH=$TREE MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TACTILE_BASE_LIBRARY=$LIB TACTILE_GRASP_CACHE=$PIPE/grasps  # the keyframes E1hold ran on
cd "$TREE" || exit 1

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
done_() { [ -e "$M/$1.done" ]; }
failed() { [ -e "$M/$1.fail" ]; }
step() {
  local name=$1; shift
  if done_ "$name"; then log "skip $name (done)"; return 0; fi
  rm -f "$M/$name.fail"
  log "start $name: $*"
  local rc=0
  "$@" >> "$OUT/logs/$name.log" 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then touch "$M/$name.done"; log "done $name"; return 0; fi
  touch "$M/$name.fail"; log "FAIL $name (exit $rc), see logs/$name.log"; return "$rc"
}
wait_file() { until [ -e "$1" ] || { [ -n "${2:-}" ] && [ -e "$2" ]; }; do sleep 120; done; [ -e "$1" ]; }

# ------------------------------------------------------------------------------------------------ the CPU track
sweep() {  # sweep NAME --policy N=P ...
  local name=$1; shift
  step "sweep_$name" "$PY_VAL" -m tactile_sim.experiments.residual_experiment run --stage "$STAGE" "$@" \
    --workers "$WORKERS" --root "$OUT/e"
}
report() {  # report NAME: the tables and figures of what has run
  step "report_$1" "$PY_RL" -m tactile_sim.experiments.friction_study --root "$OUT/e" --out "$OUT/reports/$1" \
    --stage "$STAGE"
}

cpu_track() {
  local pol=(--policy "s250=$S250")
  sweep zero_s250 "${pol[@]}"
  report a_zero_s250
  if wait_file "$PIPE/markers/export_s1k.done" "$PIPE/markers/export_s1k.fail"; then
    pol+=(--policy "s1k=$S1K")
    sweep s1k "${pol[@]}"
    report b_s1k
  fi
  report final
}

# ------------------------------------------------------------------------------------------------ main
if [ -e "$OUT/study.pid" ] && [ "$(cat "$OUT/study.pid")" != "$$" ] &&
   grep -qa run_friction_study "/proc/$(cat "$OUT/study.pid")/cmdline" 2>/dev/null; then
  echo "the study is already running (pid $(cat "$OUT/study.pid"))" >&2
  exit 1
fi
echo $$ > "$OUT/study.pid"
rm -f "$OUT/STUDY_DONE"
log "study start: tree $TREE ($(cat "$TREE/COMMIT" 2>/dev/null || echo unknown commit)), out $OUT, $WORKERS CPU workers, \
stage $STAGE"
cpu_track
log "STUDY_DONE"
echo "STUDY_DONE $(date -Is)" > "$OUT/STUDY_DONE"
