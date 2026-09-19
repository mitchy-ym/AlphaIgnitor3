#!/usr/bin/env bash
# ==============================================================================
# AlphaIgnitor3 - Daily Pipeline Automation Script
# ==============================================================================

set -euo pipefail

# Directory and Environment Setup
PROJECT_DIR="/home/yuichi/workspace/AlphaIgnitor3"
VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"
LOG_DIR="${PROJECT_DIR}/log"
LOCK_FILE="/tmp/alphaignitor3_daily.lock"
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/daily_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

# Prevent concurrent execution
exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Another instance of AlphaIgnitor3 daily pipeline is already running." >&2
    exit 1
fi

echo "================================================================================" | tee -a "${LOG_FILE}"
echo "🚀 AlphaIgnitor3 Daily Pipeline Run - ${TIMESTAMP}" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"

# Execute Daily Pipeline
if "${VENV_PYTHON}" main.py run-daily --config config/prod.yaml 2>&1 | tee -a "${LOG_FILE}"; then
    echo "================================================================================" | tee -a "${LOG_FILE}"
    echo "✅ Pipeline completed successfully at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"
    echo "================================================================================" | tee -a "${LOG_FILE}"

    # ==============================================================================
    # Portal Generation & NAS Auto Sync (ASUSTOR Drivestor 2)
    # ==============================================================================
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🌐 Generating portal index.html..." | tee -a "${LOG_FILE}"
    "${VENV_PYTHON}" "${PROJECT_DIR}/scripts/generate_portal.py" --report-dir "${PROJECT_DIR}/report" >> "${LOG_FILE}" 2>&1 || true

    NAS_TARGET_DIR="/mnt/nas_web"
    if ! mountpoint -q "${NAS_TARGET_DIR}" && mountpoint -q "/mnt/nas_report"; then
        NAS_TARGET_DIR="/mnt/nas_report"
    fi

    if mountpoint -q "${NAS_TARGET_DIR}"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 Syncing reports to NAS Web root (${NAS_TARGET_DIR})..." | tee -a "${LOG_FILE}"
        if rsync -rtv --update --no-p --no-o --no-g "${PROJECT_DIR}/report/" "${NAS_TARGET_DIR}/" >> "${LOG_FILE}" 2>&1; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ NAS sync completed successfully." | tee -a "${LOG_FILE}"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ WARNING: Failed to sync reports to NAS. Check network/mount." | tee -a "${LOG_FILE}"
        fi
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ WARNING: NAS mount point (${NAS_TARGET_DIR}) is not mounted. Skipping NAS sync." | tee -a "${LOG_FILE}"
    fi

    # ==============================================================================
    # Log Retention Cleanup (Keep last 30 days)
    # ==============================================================================
    find "${LOG_DIR}" -name "daily_*.log" -type f -mtime +30 -delete 2>/dev/null || true

    exit 0
else
    EXIT_CODE=$?
    echo "================================================================================" | tee -a "${LOG_FILE}"
    echo "❌ Pipeline failed with exit code ${EXIT_CODE} at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"
    echo "================================================================================" | tee -a "${LOG_FILE}"
    exit ${EXIT_CODE}
fi
