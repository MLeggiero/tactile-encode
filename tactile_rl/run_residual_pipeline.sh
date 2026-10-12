#!/usr/bin/env bash
# The residual's whole pipeline on the YCB hammer with the 40-D action, unattended (tactile_rl/README.md, "The 40-D
# residual on the YCB hammer"). Two tracks, each step logged and marked done (a rerun skips finished steps, resumes
# an interrupted teacher or student from its last checkpoint):
#
#   CPU: waits for $CPU_GATE_MARK in $CPU_GATE_LOG (others' CPU jobs; empty: no wait), then the CPU tests, the base
#        library (training venv) -> E1hold's zero residual (validator venv) -> each student's E1hold as its ONNX graph
#        appears
#   GPU: waits for $GATE_MARK in $GATE_LOG (another job's GPU block; empty: no wait), then the GPU tests and the load
#        generator's band check, then (once the library exists) a throughput check and per rate in $RATES: teacher
#        (Tactile-Residual C0-C3) -> DAgger student -> ONNX export with parity; then the GPU evaluation at matched
#        hold levels
#
# then the reports and $OUT/PIPELINE_DONE. Launch detached, e.g.
#   (setsid nohup tactile_rl/run_residual_pipeline.sh > ~/residual40/pipeline.out 2>&1 < /dev/null &)
# Everything is configured by environment variables or $PIPELINE_ENV (defaults: g1's layout).
set -u

# settings that must outlive a relaunch (after a reboot) live in this file, as VAR=value lines
PIPELINE_ENV=${PIPELINE_ENV:-$HOME/residual40/pipeline.env}
# shellcheck disable=SC1090
[ -f "$PIPELINE_ENV" ] && . "$PIPELINE_ENV"

TREE=${TREE:-$HOME/residual40/tree}          # a copy of this repository (git archive of the branch)
OUT=${OUT:-$HOME/residual40/runs}
PY_RL=${PY_RL:-$HOME/rl-env/.venv-rl/bin/python}           # torch, mjlab 1.6.0 (MuJoCo / MuJoCo Warp 3.11)
PY_VAL=${PY_VAL:-$HOME/tactile-encode/.venv/bin/python}    # the 8 kHz validator (MuJoCo 3.14)
GATE_LOG=${GATE_LOG-$HOME/tactile_vla/v5_pipeline.log}   # the GPU's gate (set empty: none)
GATE_MARK=${GATE_MARK:-V5_PIPELINE_DONE}
CPU_GATE_LOG=${CPU_GATE_LOG-}                            # the CPU-heavy steps' gate (empty: none)
CPU_GATE_MARK=${CPU_GATE_MARK:-SAMPLE_DONE}
LIB_EPISODES=${LIB_EPISODES:-120}
LIB_STRIKES=${LIB_STRIKES:-5}
CPU_WORKERS=${CPU_WORKERS:-2}                 # while the gate is closed (others' CPU jobs run)
CPU_WORKERS_FREE=${CPU_WORKERS_FREE:-2}       # once it is open (g1 is shared: 2 unless told otherwise)
RATES=${RATES:-"250 1000"}
NUM_ENVS=${NUM_ENVS:-4096}
TEACHER_ITERS=${TEACHER_ITERS:-1000}
STUDENT_ITERS=${STUDENT_ITERS:-300}
GPU_EVAL=${GPU_EVAL:-1}
E1_STAGE=${E1_STAGE:-E1hold}                  # hold_smoke: one seed, two strikes, 4 kHz (a dry run)
RUN_TESTS=${RUN_TESTS:-1}
THROUGHPUT_ENVS=${THROUGHPUT_ENVS:-"1024 4096"}
BAND_ENVS=${BAND_ENVS:-1024}
EVAL_ENVS=${EVAL_ENVS:-1024}
RECORD=${RECORD:-64x160}                     # the ONNX parity check: envs x steps of the student playing C3

LIB=$OUT/lib/hammer_ycb_40d.npz
M=$OUT/markers
LOG=$OUT/pipeline.log
mkdir -p "$OUT"/{lib,logs,markers,rsl_rl,students,export,e1,reports}
export PYTHONPATH=$TREE MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TACTILE_BASE_LIBRARY=$LIB
# the grasp keyframes this code synthesizes, shared by both venvs (not whatever ~/.cache holds under the same key)
export TACTILE_GRASP_CACHE=${TACTILE_GRASP_CACHE:-$OUT/grasps}
cd "$TREE" || exit 1

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
done_() { [ -e "$M/$1.done" ]; }
failed() { [ -e "$M/$1.fail" ]; }
# step NAME CMD...: run once, its output in logs/NAME.log, a .done or .fail marker
step() {
  local name=$1; shift
  if done_ "$name"; then log "skip $name (done)"; return 0; fi
  rm -f "$M/$name.fail"
  log "start $name: $*"
  if "$@" >> "$OUT/logs/$name.log" 2>&1; then touch "$M/$name.done"; log "done $name"; return 0; fi
  local rc=$?
  touch "$M/$name.fail"; log "FAIL $name (exit $rc), see logs/$name.log"; return $rc
}
# wait_for NAME: until NAME is done (0) or failed (1)
wait_for() { until done_ "$1" || failed "$1"; do sleep 120; done; done_ "$1"; }
# a gate opens when its log holds its mark (an empty log name: always open); once open it stays open (a marker)
open_gate() {  # open_gate NAME LOG MARK
  done_ "gate_$1" && return 0
  if [ -z "$2" ] || grep -q "$3" "$2" 2>/dev/null; then touch "$M/gate_$1.done"; return 0; fi
  return 1
}
gate_open() { open_gate gpu "$GATE_LOG" "$GATE_MARK"; }
cpu_gate_open() { open_gate cpu "$CPU_GATE_LOG" "$CPU_GATE_MARK"; }
cpu_workers() { if gate_open; then echo "$CPU_WORKERS_FREE"; else echo "$CPU_WORKERS"; fi; }
tag() { if [ "$1" = 1000 ]; then echo 1k; else echo "$1"; fi; }

# ------------------------------------------------------------------------------------------------ the CPU track
e1() {  # e1 NAME [--policy NAME=PATH ...]: E1hold (20 seeds x 4 holds) in the validator
  local name=$1; shift
  step "e1_$name" "$PY_VAL" -m tactile_sim.experiments.residual_experiment run --stage "$E1_STAGE" "$@" \
    --workers "$(cpu_workers)" --root "$OUT/e1"
}

cpu_track() {
  until cpu_gate_open; do sleep 60; done
  log "CPU gate open ($CPU_GATE_MARK in ${CPU_GATE_LOG:-nothing})"
  if [ "$RUN_TESTS" = 1 ]; then
    step tests_cpu "$PY_VAL" -m pytest -q tests/test_residual_l2.py tests/test_base_sources.py \
      tests/test_wuji_drive.py tests/test_residual_experiment.py || log "CPU tests failed: going on (logs/tests_cpu.log)"
  fi
  if step library "$PY_RL" -m tactile_sim.behaviors.base_library export --episodes "$LIB_EPISODES" \
       --strikes "$LIB_STRIKES" --workers "$CPU_WORKERS" --out "$LIB"; then
    # the zero-residual base should hit nearly every swing (YCB, 2026-10-12: 8 of 8 on six randomized draws)
    "$PY_RL" -c "import numpy as np; z = np.load('$LIB'); h = z['e_hit'].astype(float); \
r = z['e_slip_rot'].astype(float); print(f'library: {len(h)} entries, hits {h.mean():.3f}, slip per swing \
{np.degrees(np.nanmean(r)):.2f} deg', 'WARNING: the base misses swings' if h.mean() < 0.8 else '')" 2>&1 |
      tee -a "$LOG"
  fi
  e1 zero
  for r in $RATES; do
    local t; t=$(tag "$r")
    if wait_for "export_s$t"; then e1 "s$t" --policy "s$t=$OUT/export/student_s$t.onnx"; fi
  done
  touch "$M/cpu_track.end"
}

# ------------------------------------------------------------------------------------------------ the GPU track
latest_teacher_run() {  # the newest run directory of the rate's teacher
  ls -d "$OUT/rsl_rl/tactile_residual_teacher_$(printf %g "$1")hz/"*"_teacher_$(tag "$1")" 2>/dev/null | tail -1
}
final_teacher() {  # the teacher's last checkpoint, from whichever run reached it
  ls "$OUT/rsl_rl/tactile_residual_teacher_$(printf %g "$1")hz/"*"_teacher_$(tag "$1")/model_$((TEACHER_ITERS - 1)).pt" \
    2>/dev/null | tail -1
}

teacher() {
  local r=$1 t; t=$(tag "$r")
  local args=(-m tactile_rl.train "Tactile-Residual-VegaWuji-${r}Hz" --env.scene.num-envs "$NUM_ENVS"
              --agent.algorithm.entropy-decay-iterations "$TEACHER_ITERS" --agent.algorithm.desired-kl 0.02
              --agent.run-name "teacher_$t" --log-root "$OUT/rsl_rl")
  local run ck k best
  # the furthest checkpoint over all of this teacher's runs (a resumed run writes into a new directory, iterations
  # counted on from the checkpoint it loaded)
  best=$(ls "$OUT/rsl_rl/tactile_residual_teacher_$(printf %g "$r")hz/"*"_teacher_$t"/model_*.pt 2>/dev/null |
         sed 's/\(.*\)\/model_\([0-9]*\)\.pt$/\2 \1/' | sort -n | tail -1)
  ck=${best%% *}
  run=${best#* }
  if [ -n "$best" ] && [ -z "$(final_teacher "$r")" ]; then
    k=$ck
    log "teacher_$t: resuming $(basename "$run") from model_$k.pt"
    # the curriculum's stage is not in the checkpoint: C3 once the run had reached it (C2's cap: iteration ~280)
    TACTILE_START_STAGE=$([ "$k" -ge 300 ] && echo 3 || echo 0) "$PY_RL" "${args[@]}" \
      --agent.max-iterations "$((TEACHER_ITERS - k))" --agent.resume True --agent.load-run "$(basename "$run")" \
      --agent.load-checkpoint "model_$k.pt"
  else
    "$PY_RL" "${args[@]}" --agent.max-iterations "$TEACHER_ITERS"
  fi
  [ -n "$(final_teacher "$r")" ]
}

student() {
  local r=$1 t; t=$(tag "$r")
  local dir=$OUT/students/s$t
  local ck resume=()
  ck=$(ls "$dir"/student_*.pt 2>/dev/null | sed 's/.*student_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
  [ -n "$ck" ] && resume=(--resume "$dir/student_$ck.pt")
  "$PY_RL" -m tactile_rl.distill --rate "$r" --teacher "$(final_teacher "$r")" --library "$LIB" --log-dir "$dir" \
    --iterations "$STUDENT_ITERS" --num-envs "$NUM_ENVS" "${resume[@]}"
}

export_student() {
  local r=$1 t; t=$(tag "$r")
  "$PY_RL" -m tactile_rl.export.onnx_export "$OUT/students/s$t/student_$((STUDENT_ITERS - 1)).pt" \
    "$OUT/export/student_s$t.onnx" --student --record "$RECORD" --library "$LIB"
}

gpu_eval() {
  local pol="" r t
  for r in $RATES; do
    t=$(tag "$r")
    done_ "export_s$t" || continue
    pol="$pol,zero@$r,t$t=$(final_teacher "$r")@$r,s$t=$OUT/students/s$t/student_$((STUDENT_ITERS - 1)).pt@$r"
  done
  [ -n "$pol" ] && "$PY_RL" -m tactile_rl.bench.residual_eval --library "$LIB" --policies "${pol#,}" \
    --holds 0.3,0.5,0.8,1.0 --envs "$EVAL_ENVS" --out "$OUT/reports/eval_gpu.json"
}

gpu_track() {
  gpu_body
  local r t
  for r in $RATES; do t=$(tag "$r"); done_ "export_s$t" || touch "$M/export_s$t.fail"; done  # releases the CPU track
}

gpu_body() {
  until gate_open; do sleep 300; done
  log "GPU gate open ($GATE_MARK in ${GATE_LOG:-nothing}): the GPU is ours"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee -a "$LOG"
  if [ "$RUN_TESTS" = 1 ]; then  # (their small libraries are recorded in-process: one CPU thread)
    step tests_gpu timeout 5400 "$PY_RL" -m pytest -q tests/rl/test_rl_env.py tests/rl/test_rl_sensors.py \
      tests/rl/test_rl_drive_parity.py tests/rl/test_rl_export.py tests/rl/test_rl_student.py \
      tests/rl/test_rl_hammer.py \
      || log "GPU tests failed: training goes on (see logs/tests_gpu.log)"
  fi
  # the load generator's band (still the light hammer's) against the YCB grip at the four hold levels: a report
  step band "$PY_RL" -m tactile_rl.bench.loadgen_calibration --envs "$BAND_ENVS" --band || true
  wait_for library || { log "no library: the GPU track stops"; return 1; }
  # shellcheck disable=SC2086  # THROUGHPUT_ENVS is a list
  step throughput "$PY_RL" -m tactile_rl.bench.throughput --task hammer --seconds 0.25 --rate 1000 250 \
    --envs $THROUGHPUT_ENVS --library "$LIB" || true
  local r t
  for r in $RATES; do
    t=$(tag "$r")
    if ! step "teacher_$t" teacher "$r"; then  # once more at half the envs (GPU memory), from its checkpoints
      local n0=$NUM_ENVS
      NUM_ENVS=$((n0 / 2))
      step "teacher_${t}_half_envs" teacher "$r" && touch "$M/teacher_$t.done"
      NUM_ENVS=$n0
      done_ "teacher_$t" || continue
    fi
    step "student_s$t" student "$r" || continue
    step "export_s$t" export_student "$r"
  done
  if [ "$GPU_EVAL" = 1 ]; then step eval_gpu gpu_eval || true; fi
}

# ------------------------------------------------------------------------------------------------ main
# one pipeline at a time: the pid file's process, if it is this script (a pid from before a reboot may be reused)
if [ -e "$OUT/pipeline.pid" ] && [ "$(cat "$OUT/pipeline.pid")" != "$$" ] &&
   grep -qa run_residual_pipeline "/proc/$(cat "$OUT/pipeline.pid")/cmdline" 2>/dev/null; then
  echo "a pipeline is already running (pid $(cat "$OUT/pipeline.pid"))" >&2
  exit 1
fi
echo $$ > "$OUT/pipeline.pid"
rm -f "$OUT/PIPELINE_DONE"
log "pipeline start: tree $TREE ($(cat "$TREE/COMMIT" 2>/dev/null || echo unknown commit)), out $OUT, rates $RATES, \
library $LIB_EPISODES x $LIB_STRIKES, GPU gate '$GATE_MARK' in '$GATE_LOG', CPU gate '$CPU_GATE_MARK' in \
'$CPU_GATE_LOG', $CPU_WORKERS CPU workers"
gpu_track &
GPU_PID=$!
cpu_track
wait "$GPU_PID"
"$PY_VAL" -m tactile_sim.experiments.residual_experiment report --stage "$E1_STAGE" --root "$OUT/e1" \
  > "$OUT/reports/e1hold.md" 2>> "$LOG"
for r in $RATES; do
  run=$(latest_teacher_run "$r")
  [ -n "$run" ] && "$PY_RL" -m tactile_rl.bench.curves "$OUT/reports/curves_$(tag "$r")" "$run:$(tag "$r")" \
    >> "$OUT/logs/curves.log" 2>&1
done
ls "$M" > "$OUT/reports/markers.txt"
log "PIPELINE_DONE"
echo "PIPELINE_DONE $(date -Is)" > "$OUT/PIPELINE_DONE"
