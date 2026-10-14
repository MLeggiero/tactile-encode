#!/usr/bin/env bash
# The decision-rate sweep on the v3 recipe, unattended and resumable: a teacher and a layout-3 student per rate, trained
# exactly as s250v3 was (tactile_rl/run_v3_retrain.sh: the same band, library, keyframes, envs and iterations), then
# each student swept in the 8 kHz validator beside the friction study's policies (stage Efric, its results root).
#
#   GPU: per rate, in $RATES' order: teacher (Tactile-Residual-VegaWuji-<rate>Hz, C0-C3, $TEACHER_ENVS envs x
#        $TEACHER_ITERS) -> student s<label>v3 (DAgger, layout 3) -> export (ONNX, parity)
#   CPU: per rate, once its export is done: the sweep with that student added; after the last, the report
#
# then $OUT/RATES_DONE. Every step leaves a marker; a rerun skips finished steps (and finished validator episodes) and
# resumes a teacher from its furthest checkpoint and a student from its last. Rates are label=hz pairs; ~30 and ~60 Hz
# are SLOW_RATES' whole-step rates (tactile_rl.tasks.residual).
# Settings: environment variables or $RATES_ENV (VAR=value lines).
set -u

RATES_ENV=${RATES_ENV:-$HOME/residual40/rates.env}
# shellcheck disable=SC1090
[ -f "$RATES_ENV" ] && . "$RATES_ENV"

TREE=${TREE:-$HOME/residual40/tree_rates}
PIPE=${PIPE:-$HOME/residual40/runs}            # the main pipeline: library, keyframes
FRIC=${FRIC:-$HOME/residual40/friction}        # the friction study: its results root (e/)
V3=${V3:-$HOME/residual40/v3}                  # the v3 retrain: its band
OUT=${OUT:-$HOME/residual40/rates}
PY_RL=${PY_RL:-$HOME/tactile-encode/.venv-rl/bin/python}
PY_VAL=${PY_VAL:-$HOME/tactile-encode/.venv/bin/python}
WORKERS=${WORKERS:-10}
STAGE=${STAGE:-Efric}
RATES=${RATES:-"30=30.075187969924812 100=100.0 500=500.0 60=59.70149253731343"}
TEACHER_ITERS=${TEACHER_ITERS:-1000}
TEACHER_ENVS=${TEACHER_ENVS:-2048}
STUDENT_ITERS=${STUDENT_ITERS:-300}
STUDENT_ENVS=${STUDENT_ENVS:-2048}
RECORD=${RECORD:-64x160}

LIB=$PIPE/lib/hammer_ycb_40d.npz
BAND=$V3/band.json
M=$OUT/markers
LOG=$OUT/rates.log

export TACTILE_BASE_LIBRARY=$LIB TACTILE_GRASP_CACHE=$PIPE/grasps TACTILE_LOADGEN_BAND=$BAND MUJOCO_GL=egl
export PYTHONPATH=$TREE
mkdir -p "$OUT"/{logs,markers,rsl_rl,students,export,reports}
cd "$TREE" || exit 1

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
done_() { [ -e "$M/$1.done" ]; }
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
wait_any() { local f; while :; do for f in "$@"; do [ -e "$f" ] && return 0; done; sleep 120; done; }
hz_name() { "$PY_RL" -c 'import sys; print(f"{float(sys.argv[1]):g}")' "$1"; }  # the task id's and run dir's rate

# ------------------------------------------------------------------------------------------------ the GPU track
run_dir() { echo "$OUT/rsl_rl/tactile_residual_teacher_$(hz_name "$1")hz"; }  # run_dir HZ
final_teacher() { ls "$(run_dir "$2")/"*"_teacher_$1v3/model_$((TEACHER_ITERS - 1)).pt" 2>/dev/null | tail -1; }

teacher() {  # teacher LABEL HZ
  local label=$1 hz=$2 best ck run stage
  local args=(-m tactile_rl.train "Tactile-Residual-VegaWuji-$(hz_name "$hz")Hz" --env.scene.num-envs "$TEACHER_ENVS"
              --agent.algorithm.entropy-decay-iterations "$TEACHER_ITERS" --agent.algorithm.desired-kl 0.02
              --agent.run-name "teacher_${label}v3" --log-root "$OUT/rsl_rl")
  [ -n "$(final_teacher "$label" "$hz")" ] && return 0
  best=$(ls "$(run_dir "$hz")/"*"_teacher_${label}v3"/model_*.pt 2>/dev/null |
         sed 's/\(.*\)\/model_\([0-9]*\)\.pt$/\2 \1/' | sort -n | tail -1)
  ck=${best%% *}
  run=${best#* }
  if [ -n "$best" ]; then
    stage=$(cat "$OUT"/logs/teacher_"$label"*.log 2>/dev/null | grep -ao "Curriculum/stage/stage: *[0-9.]*" |
            sed 's/.*: *\([0-9]*\).*/\1/' | sort -n | tail -1)
    log "teacher_$label: resuming $(basename "$run") from model_$ck.pt in stage C${stage:-0}"
    TACTILE_START_STAGE=${stage:-0} "$PY_RL" "${args[@]}" --agent.max-iterations "$((TEACHER_ITERS - ck))" \
      --agent.resume True --agent.load-run "$(basename "$run")" --agent.load-checkpoint "model_$ck.pt"
  else
    "$PY_RL" "${args[@]}" --agent.max-iterations "$TEACHER_ITERS"
  fi
  [ -n "$(final_teacher "$label" "$hz")" ]
}

student() {  # student LABEL HZ
  local dir=$OUT/students/s$1v3 ck resume=()
  ck=$(ls "$dir"/student_*.pt 2>/dev/null | sed 's/.*student_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
  [ -n "$ck" ] && resume=(--resume "$dir/student_$ck.pt")
  "$PY_RL" -m tactile_rl.distill --rate "$2" --teacher "$(final_teacher "$1" "$2")" --library "$LIB" --log-dir "$dir" \
    --iterations "$STUDENT_ITERS" --num-envs "$STUDENT_ENVS" --obs-version 3 "${resume[@]}"
}

export_student() {  # export_student LABEL
  "$PY_RL" -m tactile_rl.export.onnx_export "$OUT/students/s$1v3/student_$((STUDENT_ITERS - 1)).pt" \
    "$OUT/export/student_s$1v3.onnx" --student --record "$RECORD" --library "$LIB"
}

gpu_track() {
  local pair label hz
  log "GPU: rates $RATES"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee -a "$LOG"
  for pair in $RATES; do
    label=${pair%%=*} hz=${pair#*=}
    step "teacher_$label" teacher "$label" "$hz" || step "teacher_${label}_retry" teacher "$label" "$hz" ||
      { touch "$M/export_s${label}v3.fail"; continue; }
    step "student_s${label}v3" student "$label" "$hz" && step "export_s${label}v3" export_student "$label" ||
      touch "$M/export_s${label}v3.fail"
  done
}

# ------------------------------------------------------------------------------------------------ the CPU track
cpu_track() {
  local pair label pol=()
  for pair in $RATES; do
    label=${pair%%=*}
    wait_any "$M/export_s${label}v3.done" "$M/export_s${label}v3.fail" "$M/gpu_track.over"
    if done_ "export_s${label}v3"; then
      pol=(--policy "s${label}v3=$OUT/export/student_s${label}v3.onnx")
      step "sweep_s${label}v3" "$PY_VAL" -m tactile_sim.experiments.residual_experiment run --stage "$STAGE" \
        "${pol[@]}" --workers "$WORKERS" --root "$FRIC/e"
    fi
  done
  step report_final "$PY_RL" -m tactile_sim.experiments.friction_study --root "$FRIC/e" --out "$OUT/reports/final" \
    --stage "$STAGE"
}

# ------------------------------------------------------------------------------------------------ main
if [ -e "$OUT/rates.pid" ] && [ "$(cat "$OUT/rates.pid")" != "$$" ] &&
   grep -qa run_rate_sweep "/proc/$(cat "$OUT/rates.pid")/cmdline" 2>/dev/null; then
  echo "the rate sweep is already running (pid $(cat "$OUT/rates.pid"))" >&2
  exit 1
fi
echo $$ > "$OUT/rates.pid"
rm -f "$OUT/RATES_DONE" "$M/gpu_track.over"
log "rates start: tree $TREE ($(cat "$TREE/COMMIT" 2>/dev/null || echo unknown commit)), out $OUT, results $FRIC/e, \
$WORKERS CPU workers, teachers $TEACHER_ENVS envs x $TEACHER_ITERS, students $STUDENT_ENVS envs x $STUDENT_ITERS"
( gpu_track; touch "$M/gpu_track.over" ) &
GPU_PID=$!
cpu_track
wait "$GPU_PID"
log "RATES_DONE"
echo "RATES_DONE $(date -Is)" > "$OUT/RATES_DONE"
