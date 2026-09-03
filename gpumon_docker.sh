#!/usr/bin/env bash
# 用法：
#   ./gpumon_docker_v2.sh -- <你的命令按原样分词传入>
# 例如（保持你的命令不变）：
#   ./gpumon_docker_v2.sh -- \
#     bash -c 'cmd="$*"; "$@"; echo "+ $cmd"' -- \
#     docker run --rm --gpus "device=0" \
#       -v /root/smartiot_docker/DeDiff:/workspace -it \
#       -w /workspace gnn_image \
#       python main.py --dataset twitter

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "用法: $0 -- <CMD...>"
  exit 1
fi

# 取出命令本体（支持用 -- 作为分隔）
if [ "$1" = "--" ]; then shift; fi
CMD_ARR=("$@")

# ---- 从参数里解析 docker run 的 --gpus 和 image（尽量稳健） ----
# 找到 "docker run" 的位置
docker_i=-1; run_i=-1
for i in "${!CMD_ARR[@]}"; do
  if [[ ${CMD_ARR[$i]} == docker && $((i+1)) -lt ${#CMD_ARR[@]} && ${CMD_ARR[$((i+1))]} == run ]]; then
    docker_i=$i; run_i=$((i+1)); break
  fi
done

if (( docker_i < 0 )); then
  echo "⚠️ 未在命令中找到 'docker run'，会尝试按“最新运行的容器”兜底监控。"
fi

GPU_INDEX=0
IMAGE=""

if (( docker_i >= 0 )); then
  k=$((run_i+1))
  while (( k < ${#CMD_ARR[@]} )); do
    tok="${CMD_ARR[$k]}"

    # 解析 --gpus
    if [[ "$tok" == --gpus ]]; then
      if (( k+1 < ${#CMD_ARR[@]} )); then
        val="${CMD_ARR[$((k+1))]}"
        if [[ "$val" =~ device=([0-9]+) ]]; then GPU_INDEX="${BASH_REMATCH[1]}"; fi
      fi
      k=$((k+2)); continue
    elif [[ "$tok" == --gpus=* ]]; then
      val="${tok#--gpus=}"
      if [[ "$val" =~ device=([0-9]+) ]]; then GPU_INDEX="${BASH_REMATCH[1]}"; fi
      k=$((k+1)); continue
    fi

    # 有值的选项（跳过其参数）
    case "$tok" in
      -v|--volume|-w|--workdir|--name|-e|--env|-p|--publish|--network|--ipc|--ulimit|--shm-size|--runtime|--device|--cpus|--memory|--hostname|--mount|--dns|--label|--env-file|--pull|--gpus)
        k=$((k+2)); continue ;;
    esac

    # 第一个非 - 开头的 token 视作镜像名
    if [[ "$tok" != -* && "$tok" != "" ]]; then IMAGE="$tok"; break; fi
    k=$((k+1))
  done
fi

GPU_UUID=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader | head -n1 || true)
if [ -z "${GPU_UUID:-}" ]; then
  echo "❌ 找不到 GPU $GPU_INDEX；请检查 --gpus 设备号或 nvidia-smi"
  exit 1
fi

PEAK=0
INTERVAL="${INTERVAL:-0.2}"
CID=""

sampler() {
  local gpu_uuid="$1" image="$2"
  while :; do
    # 若还没定位到容器，按镜像找最新一个；否则沿用
    if [ -z "$CID" ]; then
      if [ -n "$image" ]; then
        CID=$(docker ps --filter "ancestor=$image" --format '{{.ID}}' | head -n1 || true)
      else
        CID=$(docker ps --format '{{.ID}}' | head -n1 || true)
      fi
    fi

    if [ -n "$CID" ]; then
      RUNNING=$(docker inspect -f '{{.State.Running}}' "$CID" 2>/dev/null || echo "false")
      if [ "$RUNNING" != "true" ]; then
        sleep "$INTERVAL"; continue
      fi
      PIDS=$(docker top "$CID" -eo pid 2>/dev/null | awk 'NR>1 {printf "%s ", $1}')
      if [ -n "${PIDS:-}" ]; then
        USED=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
          | awk -v uuid="$gpu_uuid" -v pids="$PIDS" '
              BEGIN{split(pids,a," "); for(i in a) if(a[i]!="") m[a[i]]=1}
              $1==uuid && ($2 in m) {sum += $3}
              END{print (sum==0?0:sum)}
            ')
        USED=${USED:-0}
        if [ "$USED" -gt "$PEAK" ]; then PEAK=$USED; fi
      fi
    fi
    sleep "$INTERVAL"
  done
}

# 计时 + 采样（采样在后台，命令前台正常交互）
START_NS=$(date +%s%N)
sampler "$GPU_UUID" "$IMAGE" &
SAMPLE_PID=$!

# 执行你的原命令（保持一字不改）
"${CMD_ARR[@]}"
RET=$?

# 命令结束，停止采样
kill "$SAMPLE_PID" >/dev/null 2>&1 || true
wait "$SAMPLE_PID" 2>/dev/null || true

END_NS=$(date +%s%N)
ELAPSED_MS=$(( (END_NS - START_NS)/1000000 ))

echo "GPU Index: $GPU_INDEX (UUID=$GPU_UUID)"
[ -n "${CID:-}" ] && echo "Container: $CID  Image: ${IMAGE:-unknown}"
echo "Wall time: ${ELAPSED_MS} ms"
echo "Peak GPU memory (container PIDs on this GPU): ${PEAK} MiB"

exit "$RET"