#!/bin/bash
# ==============================================================================
#  FCW System – Zenity Launcher (Jetson Nano)
#
#  Hiển thị menu 3 lựa chọn, quay lại menu sau mỗi lần chạy.
#  Thoát khi bấm Cancel hoặc đóng cửa sổ (X).
#
#  Cài đặt (chỉ chạy 1 lần):
#    chmod +x scripts/launcher.sh
#    sudo apt-get install -y zenity     # thường đã có sẵn trên Ubuntu
#
#  Chạy trực tiếp:
#    ./scripts/launcher.sh
#
#  Hoặc click đúp shortcut trên Desktop (xem FCW_System.desktop)
# ==============================================================================

# ── Đường dẫn (tự động, không hardcode) ──────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"          # .../fcw_jetson
ROOT_DIR="$(dirname "${PROJECT_DIR}")"            # .../KLTN

BIN="${PROJECT_DIR}/build/fcw_system"
MODEL="${PROJECT_DIR}/models/yolov8s_bdd100k.engine"

# Thư mục mặc định cho file picker
DIR_REALWORLD="${ROOT_DIR}/video_test"            # video quay thực tế
DIR_DATASET="${ROOT_DIR}/video_data"              # video KITTI dataset

# Tạo thư mục nếu chưa có (tránh Zenity lỗi khi --filename trỏ vào thư mục không tồn tại)
mkdir -p "${DIR_REALWORLD}" "${DIR_DATASET}"

# ── Kiểm tra phụ thuộc ────────────────────────────────────────────────────────
_check_deps() {
    local ok=1
    if ! command -v zenity &>/dev/null; then
        echo "[ERROR] Zenity chưa được cài. Chạy: sudo apt-get install -y zenity"
        ok=0
    fi
    if [ ! -f "${BIN}" ]; then
        zenity --error --title="FCW – Lỗi" \
               --text="Chưa build hệ thống!\nHãy chạy: ./scripts/build_jetson.sh" \
               --width=350 2>/dev/null || \
        echo "[ERROR] Binary không tồn tại: ${BIN}"
        ok=0
    fi
    if [ ! -f "${MODEL}" ]; then
        zenity --warning --title="FCW – Cảnh báo" \
               --text="Không tìm thấy model TensorRT:\n${MODEL}\n\nHệ thống sẽ dùng model ONNX mặc định." \
               --width=400 2>/dev/null || \
        echo "[WARN] Model không tồn tại: ${MODEL}"
        MODEL=""   # để binary tự chọn fallback
    fi
    [ "${ok}" -eq 1 ]
}

# ── Chạy hệ thống C++ ─────────────────────────────────────────────────────────
_run_fcw() {
    local input_args=("$@")
    local model_args=()
    [ -n "${MODEL}" ] && model_args=(--model "${MODEL}")

    echo ""
    echo "=========================================="
    echo "  Forward Collision Warning System"
    echo "  Nhấn 'Q' hoặc ESC để dừng"
    echo "=========================================="
    echo ""

    cd "${PROJECT_DIR}"
    "${BIN}" "${model_args[@]}" "${input_args[@]}"
    local exit_code=$?

    echo ""
    echo "[INFO] Hệ thống dừng (exit code: ${exit_code}). Quay về menu..."
    sleep 1
}

# ── Vòng lặp menu chính ───────────────────────────────────────────────────────
_check_deps || exit 1

while true; do

    # ── Menu 3 lựa chọn ──────────────────────────────────────────────────────
    CHOICE=$(zenity --list \
        --title="Hệ thống Cảnh báo Va chạm Phía trước (FCW)" \
        --text="<b>Chọn nguồn đầu vào:</b>" \
        --column="" --column="Chế độ" \
        --hide-column=1 \
        --width=460 --height=260 \
        --ok-label="Chạy" \
        --cancel-label="Thoát" \
        "1" "📷  Camera trực tiếp (CSI Live)" \
        "2" "🎬  Video thực tế  –  chọn từ thư mục video_test/" \
        "3" "📂  Video dataset  –  chọn từ thư mục video_data/" \
        2>/dev/null)

    # Cancel / đóng cửa sổ → thoát
    [ -z "${CHOICE}" ] && break

    # ── Xử lý từng lựa chọn ─────────────────────────────────────────────────
    case "${CHOICE}" in

        1)  # ── Camera CSI ──────────────────────────────────────────────────
            _run_fcw --camera 0
            ;;

        2)  # ── Video thực tế ────────────────────────────────────────────────
            VIDEO=$(zenity --file-selection \
                --title="Chọn video thực tế" \
                --file-filter="Video|*.mp4 *.avi *.mkv *.mov" \
                --filename="${DIR_REALWORLD}/" \
                --width=700 --height=450 \
                2>/dev/null)

            if [ -z "${VIDEO}" ]; then
                continue   # bấm Cancel → quay lại menu chính
            fi

            if [ ! -f "${VIDEO}" ]; then
                zenity --error --title="Lỗi" \
                       --text="File không tồn tại:\n${VIDEO}" \
                       --width=350 2>/dev/null
                continue
            fi

            _run_fcw --video "${VIDEO}"
            ;;

        3)  # ── Video dataset (KITTI) ─────────────────────────────────────────
            VIDEO=$(zenity --file-selection \
                --title="Chọn video dataset" \
                --file-filter="Video|*.mp4 *.avi *.mkv" \
                --filename="${DIR_DATASET}/" \
                --width=700 --height=450 \
                2>/dev/null)

            if [ -z "${VIDEO}" ]; then
                continue
            fi

            if [ ! -f "${VIDEO}" ]; then
                zenity --error --title="Lỗi" \
                       --text="File không tồn tại:\n${VIDEO}" \
                       --width=350 2>/dev/null
                continue
            fi

            _run_fcw --video "${VIDEO}"
            ;;

    esac

done

echo "[INFO] Launcher thoát."
