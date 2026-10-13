#!/usr/bin/env bash
# Part 2 of the friction study, unattended and resumable (tactile_rl/README.md, "The v3 retrain"): the 250 Hz teacher
# and its students retrained on the training distribution of 2026-10-13 (the hand friction drawn from 0.2-2.0 on every
# episode, the wide loads at a calibrated band, VLA-style hand commands in 40 % of the episodes, observation layout 3:
# no patch features, no forecast), then swept in the 8 kHz validator beside Part 1's policies.
#
#   GPU: waits for the main pipeline's GPU steps (its eval_gpu marker, or PIPELINE_DONE), then
#        calibrate (the wide loads at friction $CALIB_FRICTION, hold $CALIB_HOLD: band.json, read by every later step
#        through $TACTILE_LOADGEN_BAND) -> teacher_v3 (Tactile-Residual C0-C3, $TEACHER_ENVS envs) -> student s250v3
#        (DAgger, layout 3) -> export (ONNX, parity) -> [ATTRIB=1] student s250v3a (layout 2 from the same teacher:
#        the attribution) -> export
#   CPU: waits for Part 1's STUDY_DONE, then a consistency check (two of Part 1's runs, rerun from this tree, against
#        Part 1's files), the sweep with s250v3 added (Efric, Part 1's results root: its runs are not rerun), the report;
#        then the same with s250v3a.
#
# then $OUT/V3_DONE. Every step leaves a marker; a rerun skips finished steps (and finished validator episodes) and
# resumes the teacher (in the curriculum stage its log last showed) and the students from their checkpoints.
# Settings: environment variables or $V3_ENV (VAR=value lines).
set -u

V3_ENV=${V3_ENV:-$HOME/residual40/v3.env}
# shellcheck disable=SC1090
[ -f "$V3_ENV" ] && . "$V3_ENV"

TREE=${TREE:-$HOME/residual40/tree_v3}
PIPE=${PIPE:-$HOME/residual40/runs}            # the main pipeline: library, grasps, Part 1's students
FRIC=${FRIC:-$HOME/residual40/friction}        # Part 1: its results root (e/) and STUDY_DONE
OUT=${OUT:-$HOME/residual40/v3}
PY_RL=${PY_RL:-$HOME/tactile-encode/.venv-rl/bin/python}
PY_VAL=${PY_VAL:-$HOME/tactile-encode/.venv/bin/python}
WORKERS=${WORKERS:-10}
STAGE=${STAGE:-Efric}
CALIB_ENVS=${CALIB_ENVS:-2048}
CALIB_FRICTION=${CALIB_FRICTION:-0.6}
CALIB_HOLD=${CALIB_HOLD:-0.5}
TARGET_LOST=${TARGET_LOST:-0.4}
TEACHER_ITERS=${TEACHER_ITERS:-1000}
TEACHER_ENVS=${TEACHER_ENVS:-2048}
STUDENT_ITERS=${STUDENT_ITERS:-300}
STUDENT_ENVS=${STUDENT_ENVS:-2048}
RECORD=${RECORD:-64x160}
ATTRIB=${ATTRIB:-1}
GPU_WAIT=${GPU_WAIT:-1}                        # 0: the GPU is free now

LIB=$PIPE/lib/hammer_ycb_40d.npz
S250=$PIPE/export/student_s250.onnx
S1K=$PIPE/export/student_s1k.onnx
BAND=$OUT/band.json
M=$OUT/markers
LOG=$OUT/v3.log
mkdir -p "$OUT"/{logs,markers,rsl_rl,students,export,reports,consistency}
export PYTHONPATH=$TREE MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TACTILE_BASE_LIBRARY=$LIB TACTILE_GRASP_CACHE=$PIPE/grasps  # the keyframes Part 1 ran on
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

# ------------------------------------------------------------------------------------------------ the GPU track
calibrate() {
  "$PY_RL" -m tactile_rl.bench.loadgen_calibration --wide --friction "$CALIB_FRICTION" --holds "$CALIB_HOLD" \
    --envs "$CALIB_ENVS" --target-lost "$TARGET_LOST" --out "$OUT/reports/calibration_rows.json" --choose "$BAND.tmp" &&
    mv "$BAND.tmp" "$BAND"
}

run_dir() { echo "$OUT/rsl_rl/tactile_residual_teacher_250hz"; }
final_teacher() { ls "$(run_dir)/"*"_teacher_250v3/model_$((TEACHER_ITERS - 1)).pt" 2>/dev/null | tail -1; }

teacher() {
  local args=(-m tactile_rl.train Tactile-Residual-VegaWuji-250Hz --env.scene.num-envs "$TEACHER_ENVS"
              --agent.algorithm.entropy-decay-iterations "$TEACHER_ITERS" --agent.algorithm.desired-kl 0.02
              --agent.run-name teacher_250v3 --log-root "$OUT/rsl_rl")
  local best ck run stage
  [ -n "$(final_teacher)" ] && return 0  # (a rerun after its last checkpoint)
  # the furthest checkpoint over all of this teacher's runs (a resumed run writes into a new directory)
  best=$(ls "$(run_dir)/"*"_teacher_250v3"/model_*.pt 2>/dev/null |
         sed 's/\(.*\)\/model_\([0-9]*\)\.pt$/\2 \1/' | sort -n | tail -1)
  ck=${best%% *}
  run=${best#* }
  if [ -n "$best" ] && [ -z "$(final_teacher)" ]; then
    # the curriculum's stage is not in the checkpoint: the furthest one the training's logs showed
    stage=$(cat "$OUT"/logs/teacher_v3*.log 2>/dev/null | grep -ao "Curriculum/stage/stage: *[0-9.]*" |
            sed 's/.*: *\([0-9]*\).*/\1/' | sort -n | tail -1)
    log "teacher_v3: resuming $(basename "$run") from model_$ck.pt in stage C${stage:-0}"
    TACTILE_START_STAGE=${stage:-0} "$PY_RL" "${args[@]}" --agent.max-iterations "$((TEACHER_ITERS - ck))" \
      --agent.resume True --agent.load-run "$(basename "$run")" --agent.load-checkpoint "model_$ck.pt"
  else
    "$PY_RL" "${args[@]}" --agent.max-iterations "$TEACHER_ITERS"
  fi
  [ -n "$(final_teacher)" ]
}

student() {  # student NAME LAYOUT
  local dir=$OUT/students/$1 ck resume=()
  ck=$(ls "$dir"/student_*.pt 2>/dev/null | sed 's/.*student_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
  [ -n "$ck" ] && resume=(--resume "$dir/student_$ck.pt")
  "$PY_RL" -m tactile_rl.distill --rate 250 --teacher "$(final_teacher)" --library "$LIB" --log-dir "$dir" \
    --iterations "$STUDENT_ITERS" --num-envs "$STUDENT_ENVS" --obs-version "$2" "${resume[@]}"
}

export_student() {  # export_student NAME
  "$PY_RL" -m tactile_rl.export.onnx_export "$OUT/students/$1/student_$((STUDENT_ITERS - 1)).pt" \
    "$OUT/export/student_$1.onnx" --student --record "$RECORD" --library "$LIB"
}

gpu_track() {
  if [ "$GPU_WAIT" = 1 ]; then
    wait_any "$PIPE/markers/eval_gpu.done" "$PIPE/markers/eval_gpu.fail" "$PIPE/PIPELINE_DONE"
  fi
  log "GPU: free"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee -a "$LOG"
  step calibrate calibrate || { log "no band: the GPU track stops"; return 1; }
  export TACTILE_LOADGEN_BAND=$BAND
  log "band: $("$PY_VAL" -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["band"]))' "$BAND")"
  step teacher_v3 teacher || step teacher_v3_retry teacher || return 1
  step student_s250v3 student s250v3 3 && step export_s250v3 export_student s250v3
  if [ "$ATTRIB" = 1 ]; then
    step student_s250v3a student s250v3a 2 && step export_s250v3a export_student s250v3a
  fi
}

# ------------------------------------------------------------------------------------------------ the CPU track
consistency() {  # two of Part 1's runs from this tree: the zero residual and s250 (layout 2), against Part 1's files
  "$PY_VAL" - "$FRIC/e" "$OUT/consistency" "$S250" "$STAGE" <<'PY'
import json
import sys
from pathlib import Path

from tactile_sim.experiments.residual_experiment import HAMMER, make_jobs, run_all

src, dst, s250, stage = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
want = {"base@hold=1@mu=0.6", "residual:s250@hold=1@mu=0.6"}
jobs = [j for j in make_jobs(stage, {"s250": s250}, None, "hammer", HAMMER) if j.condition.name in want and j.seed == 0]
assert len(jobs) == 2, [j.condition.name for j in jobs]
run_all(jobs, dst, 2)


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k != "wall_s":
                yield from leaves(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, x


bad = 0
for j in jobs:
    a, b = (dict(leaves(json.loads(j.path(r).read_text()))) for r in (src, dst))
    diff = [k for k in set(a) | set(b) if a.get(k) != b.get(k) and not (a.get(k) != a.get(k) and b.get(k) != b.get(k))]
    worst = max((abs(a[k] - b[k]) for k in diff if isinstance(a.get(k), float) and isinstance(b.get(k), float)),
                default=0.0)
    print(f"{j.condition.name} seed {j.seed}: {len(a)} values, {len(diff)} differ (largest numeric difference {worst:g})")
    for k in sorted(diff)[:10]:
        print(f"  {k}: {a.get(k)} vs {b.get(k)}")
    bad += bool(diff)
print("CONSISTENT" if not bad else "DIFFERENT")
PY
}

sweep() {  # sweep NAME --policy N=P ...
  local name=$1; shift
  step "sweep_$name" "$PY_VAL" -m tactile_sim.experiments.residual_experiment run --stage "$STAGE" "$@" \
    --workers "$WORKERS" --root "$FRIC/e"
}
report() {
  step "report_$1" "$PY_RL" -m tactile_sim.experiments.friction_study --root "$FRIC/e" --out "$OUT/reports/$1" \
    --stage "$STAGE"
}

cpu_track() {
  wait_any "$FRIC/STUDY_DONE"
  log "CPU: Part 1's study is done"
  step consistency consistency || true
  local pol=(--policy "s250=$S250" --policy "s1k=$S1K")
  wait_any "$M/export_s250v3.done" "$M/export_s250v3.fail" "$M/gpu_track.over"
  if done_ export_s250v3; then
    pol+=(--policy "s250v3=$OUT/export/student_s250v3.onnx")
    sweep s250v3 "${pol[@]}"
    report a_s250v3
  fi
  if [ "$ATTRIB" = 1 ]; then
    wait_any "$M/export_s250v3a.done" "$M/export_s250v3a.fail" "$M/gpu_track.over"
    if done_ export_s250v3a; then
      pol+=(--policy "s250v3a=$OUT/export/student_s250v3a.onnx")
      sweep s250v3a "${pol[@]}"
    fi
  fi
  report final
}

# ------------------------------------------------------------------------------------------------ main
if [ -e "$OUT/v3.pid" ] && [ "$(cat "$OUT/v3.pid")" != "$$" ] &&
   grep -qa run_v3_retrain "/proc/$(cat "$OUT/v3.pid")/cmdline" 2>/dev/null; then
  echo "the v3 retrain is already running (pid $(cat "$OUT/v3.pid"))" >&2
  exit 1
fi
echo $$ > "$OUT/v3.pid"
rm -f "$OUT/V3_DONE" "$M/gpu_track.over"
log "v3 start: tree $TREE ($(cat "$TREE/COMMIT" 2>/dev/null || echo unknown commit)), out $OUT, results $FRIC/e, \
$WORKERS CPU workers, teacher $TEACHER_ENVS envs x $TEACHER_ITERS, students $STUDENT_ENVS envs x $STUDENT_ITERS, \
attribution $ATTRIB"
( gpu_track; touch "$M/gpu_track.over" ) &
GPU_PID=$!
cpu_track
wait "$GPU_PID"
log "V3_DONE"
echo "V3_DONE $(date -Is)" > "$OUT/V3_DONE"
