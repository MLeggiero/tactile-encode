#!/usr/bin/env bash
# The friction study, unattended and resumable (tactile_rl/README.md, "The friction study"): the hand's friction on the
# YCB hammer fixed per condition in the 8 kHz validator (residual_experiment stage Efric: mu 0.2-1.5, holds 1.0 and
# 0.5, 12 paired seeds), the zero residual and the students, with the residual's own activity; and a 250 Hz teacher
# and student retrained with the hand friction drawn from $WIDE, swept the same way.
#
#   CPU: sweep (zero, s250) -> report -> [s1k's ONNX] sweep (+ s1k) -> report -> [s250w's ONNX] sweep (+ s250w)
#   GPU: waits for the main pipeline's PIPELINE_DONE and for $OUT/gpu_code.ready, then teacher_250w (Tactile-Residual
#        C0-C3 with $TACTILE_HAND_FRICTION=$WIDE, $TEACHER_ENVS envs) -> DAgger student s250w -> its ONNX export
#
# then the final report (friction_study: report.md and the figures) and $OUT/STUDY_DONE. Every step leaves a marker; a
# rerun skips finished steps (and finished validator episodes) and resumes the teacher / student from checkpoints.
# Settings: environment variables or $STUDY_ENV (VAR=value lines).
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
WIDE=${WIDE:-0.2,2.0}                          # the retrained teacher's hand friction range (ResidualDRCfg's 1.1-2.0)
TEACHER_ITERS=${TEACHER_ITERS:-1000}
TEACHER_ENVS=${TEACHER_ENVS:-2048}             # as the s250 teacher ran (4096 ran out of GPU memory)
STUDENT_ITERS=${STUDENT_ITERS:-300}
STUDENT_ENVS=${STUDENT_ENVS:-4096}
RECORD=${RECORD:-64x160}
GPU=${GPU:-1}

LIB=$PIPE/lib/hammer_ycb_40d.npz
S250=$PIPE/export/student_s250.onnx
S1K=$PIPE/export/student_s1k.onnx
S250W=$OUT/export/student_s250w.onnx
M=$OUT/markers
LOG=$OUT/study.log
mkdir -p "$OUT"/{logs,markers,e,rsl_rl,students,export,reports}
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
  if [ "$GPU" = 1 ] && wait_file "$M/export_s250w.done" "$M/export_s250w.fail"; then
    pol+=(--policy "s250w=$S250W")
    sweep s250w "${pol[@]}"
  fi
  report final
}

# ------------------------------------------------------------------------------------------------ the GPU track
final_teacher() {
  ls "$OUT/rsl_rl/tactile_residual_teacher_250hz/"*"_teacher_250w/model_$((TEACHER_ITERS - 1)).pt" 2>/dev/null | tail -1
}

teacher() {
  local args=(-m tactile_rl.train Tactile-Residual-VegaWuji-250Hz --env.scene.num-envs "$TEACHER_ENVS"
              --agent.algorithm.entropy-decay-iterations "$TEACHER_ITERS" --agent.algorithm.desired-kl 0.02
              --agent.run-name teacher_250w --log-root "$OUT/rsl_rl")
  local best ck run
  best=$(ls "$OUT/rsl_rl/tactile_residual_teacher_250hz/"*"_teacher_250w"/model_*.pt 2>/dev/null |
         sed 's/\(.*\)\/model_\([0-9]*\)\.pt$/\2 \1/' | sort -n | tail -1)
  ck=${best%% *}
  run=${best#* }
  if [ -n "$best" ] && [ -z "$(final_teacher)" ]; then
    log "teacher_250w: resuming $(basename "$run") from model_$ck.pt"
    TACTILE_HAND_FRICTION=$WIDE TACTILE_START_STAGE=$([ "$ck" -ge 300 ] && echo 3 || echo 0) "$PY_RL" "${args[@]}" \
      --agent.max-iterations "$((TEACHER_ITERS - ck))" --agent.resume True --agent.load-run "$(basename "$run")" \
      --agent.load-checkpoint "model_$ck.pt"
  else
    TACTILE_HAND_FRICTION=$WIDE "$PY_RL" "${args[@]}" --agent.max-iterations "$TEACHER_ITERS"
  fi
  [ -n "$(final_teacher)" ]
}

student() {
  local dir=$OUT/students/s250w ck resume=()
  ck=$(ls "$dir"/student_*.pt 2>/dev/null | sed 's/.*student_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
  [ -n "$ck" ] && resume=(--resume "$dir/student_$ck.pt")
  TACTILE_HAND_FRICTION=$WIDE "$PY_RL" -m tactile_rl.distill --rate 250 --teacher "$(final_teacher)" --library "$LIB" \
    --log-dir "$dir" --iterations "$STUDENT_ITERS" --num-envs "$STUDENT_ENVS" "${resume[@]}"
}

export_student() {
  TACTILE_HAND_FRICTION=$WIDE "$PY_RL" -m tactile_rl.export.onnx_export \
    "$OUT/students/s250w/student_$((STUDENT_ITERS - 1)).pt" "$S250W" --student --record "$RECORD" --library "$LIB"
}

gpu_track() {
  wait_file "$PIPE/PIPELINE_DONE"
  wait_file "$OUT/gpu_code.ready"
  log "GPU: the main pipeline is done; retraining at hand friction $WIDE"
  step teacher_250w teacher && step student_s250w student && step export_s250w export_student
  done_ export_s250w || touch "$M/export_s250w.fail"  # releases the CPU track
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
stage $STAGE, GPU $GPU (friction $WIDE, teacher $TEACHER_ENVS envs x $TEACHER_ITERS)"
GPU_PID=""
if [ "$GPU" = 1 ]; then
  gpu_track &
  GPU_PID=$!
fi
cpu_track
[ -n "$GPU_PID" ] && wait "$GPU_PID"
log "STUDY_DONE"
echo "STUDY_DONE $(date -Is)" > "$OUT/STUDY_DONE"
