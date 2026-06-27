#!/bin/bash
# Interactive launcher for the skills-extraction pipeline (titan3).
#
# Asks for input file, sample size, run id / resume, endpoints, ports, and
# model names, then launches the full two-phase run (Qwen extract -> swap to
# Mistral-Nemo for verify/classify) under nohup so it survives disconnect.
#
# Usage:
#   ./launch.sh
#
# Servers are assumed already running. Start Qwen first:  ~/startQwen 8
set -euo pipefail

BASE_DIR="$HOME/skills_extraction"

# Defaults — input file, output dir, and log dir are independently located so
# different programs can own different locations.
DEF_INPUT="../data_files/SampleJobs.json"
DEF_INPUT_DIR="../data_files"
DEF_OUTPUT="../data_files"
DEF_LOGS="./logs"
DEF_SAMPLE="0"
DEF_ENDPOINTS="8"
DEF_BASE_PORT="8000"
DEF_EXTRACTOR="Qwen/Qwen3-14B"
DEF_VERIFIER="mistralai/Mistral-Nemo-Instruct-2407"

cd "$BASE_DIR"

echo "=========================================="
echo "  Skills Extraction — Interactive Launcher"
echo "=========================================="
echo ""

# --- ask: input file (scrollable picker over .json files in the data dir) ---
shopt -s nullglob
json_files=( "$DEF_INPUT_DIR"/*.json )
shopt -u nullglob

if [ ${#json_files[@]} -gt 0 ]; then
    echo "Input JSON file — choose from $DEF_INPUT_DIR:"
    PS3="Select input file number: "
    select choice in "${json_files[@]}" "Enter a custom path"; do
        if [ -z "$choice" ]; then
            echo "  ! invalid selection (enter a listed number)"
            continue
        fi
        if [ "$choice" = "Enter a custom path" ]; then
            read -r -p "  Path to input JSON: " INPUT
        else
            INPUT="$choice"
        fi
        if [ -f "$INPUT" ]; then
            break
        fi
        echo "  ! file not found: $INPUT"
    done
else
    echo "  (no .json files found in $DEF_INPUT_DIR)"
    while true; do
        read -r -p "Path to input JSON [$DEF_INPUT]: " INPUT
        INPUT="${INPUT:-$DEF_INPUT}"
        [ -f "$INPUT" ] && break
        echo "  ! file not found: $INPUT"
    done
fi
echo "  -> using: $INPUT"

# --- ask: sample size ---
read -r -p "Sample size, 0 = all jobs [$DEF_SAMPLE]: " SAMPLE
SAMPLE="${SAMPLE:-$DEF_SAMPLE}"

# --- ask: output directory (result files; checkpoints live here too) ---
read -r -p "Output directory for result files [$DEF_OUTPUT]: " OUTPUT_DIR
OUTPUT_DIR="${OUTPUT_DIR:-$DEF_OUTPUT}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"   # resolve to absolute
RUN_CKPT_DIR="$OUTPUT_DIR/checkpoints"

# --- ask: log directory (run log + generated runner script; separate from data) ---
read -r -p "Log directory (run log + runner script) [$DEF_LOGS]: " LOG_DIR
LOG_DIR="${LOG_DIR:-$DEF_LOGS}"
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"   # resolve to absolute

# --- ask: new run or resume ---
RERUN_FROM=""
RETRY_STAGE1="no"
read -r -p "Run mode — (n)ew or (r)esume an existing run-id? [n]: " RUN_MODE
RUN_MODE="${RUN_MODE:-n}"
if [[ "$RUN_MODE" =~ ^[Rr] ]]; then
    while true; do
        read -r -p "  Existing run-id to resume: " RUN_ID
        if [ -n "$RUN_ID" ]; then
            break
        fi
        echo "  ! run-id cannot be empty"
    done
    echo "  Resume options:"
    echo "    1) Resume as-is (continue from last checkpoint)"
    echo "    2) Rerun from a stage (--rerun-from)"
    echo "    3) Retry only stage1 errors (--retry-stage1-errors)"
    read -r -p "  Choose [1]: " RESUME_OPT
    RESUME_OPT="${RESUME_OPT:-1}"
    case "$RESUME_OPT" in
        2)
            read -r -p "    Rerun from which stage (stage1/stage2/stage3/stage4): " RERUN_FROM
            ;;
        3)
            RETRY_STAGE1="yes"
            ;;
    esac
else
    RUN_ID="launch_$(date +%Y%m%d_%H%M%S)"
fi

# --- ask: endpoints / ports ---
read -r -p "Number of vLLM endpoints [$DEF_ENDPOINTS]: " ENDPOINTS
ENDPOINTS="${ENDPOINTS:-$DEF_ENDPOINTS}"
read -r -p "vLLM base port [$DEF_BASE_PORT]: " BASE_PORT
BASE_PORT="${BASE_PORT:-$DEF_BASE_PORT}"

# --- ask: model names ---
read -r -p "Extractor model (stage 1) [$DEF_EXTRACTOR]: " EXTRACTOR
EXTRACTOR="${EXTRACTOR:-$DEF_EXTRACTOR}"
read -r -p "Verifier/classifier model (stages 2-4) [$DEF_VERIFIER]: " VERIFIER
VERIFIER="${VERIFIER:-$DEF_VERIFIER}"

# --- derived ports list ---
LAST_PORT=$((BASE_PORT + ENDPOINTS - 1))
PORTS=$(seq "$BASE_PORT" "$LAST_PORT")

# --- summary ---
TS="$(date +%Y%m%d_%H%M%S)"
RUNNER="$LOG_DIR/run_${RUN_ID}.sh"
LOGFILE="$LOG_DIR/run_${RUN_ID}_${TS}.log"

echo ""
echo "=========================================="
echo "  Review settings"
echo "=========================================="
echo "  Input file        : $INPUT"
echo "  Sample size       : $SAMPLE  (0 = all)"
echo "  Run id            : $RUN_ID"
if [[ "$RUN_MODE" =~ ^[Rr] ]]; then
    echo "  Mode              : RESUME"
    [ -n "$RERUN_FROM" ]        && echo "  Rerun from        : $RERUN_FROM"
    [ "$RETRY_STAGE1" = "yes" ] && echo "  Retry stage1 errs : yes"
else
    echo "  Mode              : NEW"
fi
echo "  Endpoints         : $ENDPOINTS  (ports $BASE_PORT-$LAST_PORT)"
echo "  Extractor model   : $EXTRACTOR"
echo "  Verify/classify   : $VERIFIER"
echo "  Output dir        : $OUTPUT_DIR"
echo "  Log dir           : $LOG_DIR"
echo "  Runner script     : $RUNNER"
echo "  Log file          : $LOGFILE"
echo ""

read -r -p "Launch this run under nohup? [y/N]: " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy] ]]; then
    echo "Aborted. Nothing launched."
    exit 0
fi

# --- build extra flags for the extraction phase ---
EXTRA_FLAGS=""
[ -n "$RERUN_FROM" ]        && EXTRA_FLAGS="$EXTRA_FLAGS --rerun-from $RERUN_FROM"
[ "$RETRY_STAGE1" = "yes" ] && EXTRA_FLAGS="$EXTRA_FLAGS --retry-stage1-errors"
SAMPLE_FLAG=""
[ "$SAMPLE" != "0" ] && SAMPLE_FLAG="--sample $SAMPLE"

# --- write the self-contained runner (executed under nohup) ---
cat > "$RUNNER" <<RUNNER_EOF
#!/bin/bash
# Auto-generated by launch.sh on $TS — run-id $RUN_ID
set -euo pipefail
cd "$BASE_DIR"

INPUT="$INPUT"
OUTPUT_DIR="$OUTPUT_DIR"
RUN_ID="$RUN_ID"
ENDPOINTS="$ENDPOINTS"
BASE_PORT="$BASE_PORT"
EXTRACTOR="$EXTRACTOR"
VERIFIER="$VERIFIER"
PORTS="$PORTS"
EXTRA_FLAGS="$EXTRA_FLAGS"
SAMPLE_FLAG="$SAMPLE_FLAG"
STAGE1_CKPT="$RUN_CKPT_DIR/\${RUN_ID}_stage1_extracted.jsonl"

echo "=== Skills extraction run \$RUN_ID — \$(date) ==="

# Is stage 1 already complete for this run-id? (resume fast-path)
stage1_done() {
    [ -f "\$STAGE1_CKPT" ] || return 1
    tail -1 "\$STAGE1_CKPT" | python3 -c "import sys,json; exit(0 if json.load(sys.stdin).get('_complete') else 1)" 2>/dev/null
}

# Detect the model currently loaded on the base port.
loaded_model() {
    curl -s "http://localhost:\${BASE_PORT}/v1/models" \\
        | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo ""
}

verify_endpoints() {
    for port in \$PORTS; do
        if ! curl -s "http://localhost:\${port}/v1/models" > /dev/null 2>&1; then
            echo "ERROR: vLLM not responding on port \$port"
            exit 1
        fi
    done
}

if stage1_done; then
    echo "=== Stage 1 already complete for \$RUN_ID — skipping extraction phase ==="
else
    echo ""
    echo "=== Phase 1: verifying Qwen endpoints ==="
    verify_endpoints
    MODEL=\$(loaded_model)
    echo "Model on port \$BASE_PORT: \$MODEL"
    case "\$MODEL" in
        *[Qq]wen*) : ;;
        *) echo "ERROR: expected Qwen on the endpoints for extraction; found '\$MODEL'. Start it: ~/startQwen \$ENDPOINTS"; exit 1 ;;
    esac

    echo ""
    echo "=== Stages 0-1: Extraction (\$EXTRACTOR) ==="
    conda run --no-capture-output -n skills python -m skills_extraction \\
      --input "\$INPUT" \\
      --output-dir "\$OUTPUT_DIR" \\
      --run-id "\$RUN_ID" \\
      \$SAMPLE_FLAG \\
      \$EXTRA_FLAGS \\
      --vllm \\
      --vllm-host localhost \\
      --vllm-base-port "\$BASE_PORT" \\
      --vllm-num-endpoints "\$ENDPOINTS" \\
      --extractor-model "\$EXTRACTOR" \\
      --no-verifier \\
      --no-requirement-classifier \\
      --no-hardsoft-classifier

    echo ""
    echo "=== Stage 1 complete. Swapping to \$VERIFIER ==="
    ~/stopModel
    echo "Waiting for extraction model to shut down..."
    sleep 10
    ~/startMistral "\$ENDPOINTS"
fi

echo ""
echo "=== Phase 2: ensuring verify/classify model is ready ==="
MODEL=\$(loaded_model)
case "\$MODEL" in
    *[Mm]istral*) echo "Mistral already loaded: \$MODEL" ;;
    *)
        echo "Loading \$VERIFIER ..."
        ~/stopModel || true
        sleep 10
        ~/startMistral "\$ENDPOINTS"
        ;;
esac

echo "Waiting for \$VERIFIER to load..."
for i in \$(seq 1 60); do
    if curl -s "http://localhost:\${BASE_PORT}/v1/models" > /dev/null 2>&1; then
        echo "Ready after \${i}0 seconds"
        break
    fi
    sleep 10
done
verify_endpoints

echo ""
echo "=== Stages 2-4: Verify + Classify (\$VERIFIER) — resuming \$RUN_ID ==="
conda run --no-capture-output -n skills python -m skills_extraction \\
  --input "\$INPUT" \\
  --output-dir "\$OUTPUT_DIR" \\
  --run-id "\$RUN_ID" \\
  --vllm \\
  --vllm-host localhost \\
  --vllm-base-port "\$BASE_PORT" \\
  --vllm-num-endpoints "\$ENDPOINTS" \\
  --extractor-model "\$EXTRACTOR" \\
  --verifier-model "\$VERIFIER" \\
  --requirement-model "\$VERIFIER" \\
  --hardsoft-model "\$VERIFIER"

echo ""
echo "=== Pipeline complete — \$(date) — run \$RUN_ID ==="
SKILLS_OUT_DIR="\$OUTPUT_DIR" ./check_status.sh "\$RUN_ID" 2>/dev/null || true
RUNNER_EOF

chmod +x "$RUNNER"

# --- launch under nohup ---
nohup bash "$RUNNER" > "$LOGFILE" 2>&1 &
PID=$!

echo ""
echo "=========================================="
echo "  Launched — run-id $RUN_ID  (pid $PID)"
echo "=========================================="
echo "  Follow log : tail -f $LOGFILE"
echo "  Status     : ./check_status.sh $RUN_ID"
echo "  Stop       : kill $PID"
echo ""
