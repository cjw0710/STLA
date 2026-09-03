#!/usr/bin/env bash
# 监测所有 GPU 在指定时间窗口内（默认1小时）的最大显存占用
# 依赖：nvidia-smi（NVIDIA 驱动环境）
#
# 用法：
#   ./gpumon_peak_1h.sh                     # 监测1小时，2秒采样
#   ./gpumon_peak_1h.sh -d 600 -i 1         # 监测10分钟，每秒采样
#   ./gpumon_peak_1h.sh --csv -o peak.csv   # CSV 输出到文件
#
# 参数：
#   -d, --duration SECONDS   总时长（秒），默认 4000
#   -i, --interval SECONDS   采样间隔（秒），默认 2
#   -o, --output PATH        输出文件路径（默认仅打印到控制台）
#       --csv                以 CSV 格式输出
#   -h, --help               显示帮助

set -euo pipefail

duration=4000
interval=2
outfile=""
csv=0

print_help() {
  sed -n '1,40p' "$0"
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "未找到 nvidia-smi，请在有 NVIDIA 驱动的环境中运行。" >&2
  exit 1
fi

# 解析参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--duration)
      [[ $# -ge 2 ]] || { echo "--duration 需要数值" >&2; exit 1; }
      duration="$2"; shift 2;;
    -i|--interval)
      [[ $# -ge 2 ]] || { echo "--interval 需要数值" >&2; exit 1; }
      interval="$2"; shift 2;;
    -o|--output)
      [[ $# -ge 2 ]] || { echo "--output 需要路径" >&2; exit 1; }
      outfile="$2"; shift 2;;
    --csv)
      csv=1; shift;;
    -h|--help)
      print_help; exit 0;;
    *)
      echo "未知参数: $1" >&2; print_help; exit 1;;
  esac
done

# 获取 GPU 基本信息
mapfile -t gpu_info < <(nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits)
ngpus=${#gpu_info[@]}

if (( ngpus == 0 )); then
  echo "未检测到 GPU。" >&2
  exit 1
fi

declare -a idx_arr uuid_arr name_arr total_arr
declare -a peak_arr peak_ts_arr

for line in "${gpu_info[@]}"; do
  idx=$(echo "$line"   | cut -d, -f1 | xargs)
  uuid=$(echo "$line"  | cut -d, -f2 | xargs)
  name=$(echo "$line"  | cut -d, -f3 | xargs)
  total=$(echo "$line" | cut -d, -f4 | xargs)

  idx_arr[idx]="$idx"
  uuid_arr[idx]="$uuid"
  name_arr[idx]="$name"
  total_arr[idx]="$total"
  peak_arr[idx]=0
  peak_ts_arr[idx]=""
done

stop=0
trap 'stop=1' INT TERM

start_ts=$(date +%s)
end_ts=$(( start_ts + duration ))

while :; do
  now=$(date +%s)
  if (( now >= end_ts || stop == 1 )); then
    break
  fi

  ts_human=$(date +%Y-%m-%dT%H:%M:%S%z)
  mapfile -t used_list < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

  # 假设 nvidia-smi 输出顺序与 index 一致（通常如此）
  for i in "${!used_list[@]}"; do
    used=$(echo "${used_list[$i]}" | xargs)
    # 有些环境索引可能不是 0..N-1 连续，使用 idx_arr 的现有索引
    if [[ -n "${peak_arr[$i]:-}" ]]; then
      if (( used > ${peak_arr[$i]:-0} )); then
        peak_arr[$i]=$used
        peak_ts_arr[$i]="$ts_human"
      fi
    else
      # 若数组未初始化（极少见），初始化后赋值
      peak_arr[$i]=$used
      peak_ts_arr[$i]="$ts_human"
    fi
  done

  sleep "$interval"
done

# 输出结果
emit_text() {
  printf "Monitoring window: %s seconds, interval: %s seconds\n" "$duration" "$interval"
  printf "%-5s %-36s %-24s %10s %12s %8s %s\n" "IDX" "UUID" "NAME" "TOTAL(MiB)" "PEAK(MiB)" "%" "PEAK_TIME"
  for i in "${!idx_arr[@]}"; do
    idx=${idx_arr[$i]}
    uuid=${uuid_arr[$i]}
    name=${name_arr[$i]}
    total=${total_arr[$i]}
    peak=${peak_arr[$i]:-0}
    pct=0
    if [[ -n "$total" && "$total" =~ ^[0-9]+$ && "$peak" =~ ^[0-9]+$ && $total -gt 0 ]]; then
      pct=$(( 100 * peak / total ))
    fi
    printf "%-5s %-36s %-24s %10s %12s %8s %s\n" "$idx" "$uuid" "${name:0:24}" "$total" "$peak" "$pct" "${peak_ts_arr[$i]}"
  done
}

emit_csv() {
  printf "index,uuid,name,memory_total_MiB,peak_used_MiB,peak_percent,peak_time\n"
  for i in "${!idx_arr[@]}"; do
    idx=${idx_arr[$i]}
    uuid=${uuid_arr[$i]}
    name=${name_arr[$i]}
    total=${total_arr[$i]}
    peak=${peak_arr[$i]:-0}
    pct=0
    if [[ -n "$total" && "$total" =~ ^[0-9]+$ && "$peak" =~ ^[0-9]+$ && $total -gt 0 ]]; then
      pct=$(( 100 * peak / total ))
    fi
    printf "%s,%s,%s,%s,%s,%s,%s\n" "$idx" "$uuid" "$name" "$total" "$peak" "$pct" "${peak_ts_arr[$i]}"
  done
}

if (( csv == 1 )); then
  if [[ -n "$outfile" ]]; then
    emit_csv >"$outfile"
    echo "CSV 已写入: $outfile"
  else
    emit_csv
  fi
else
  if [[ -n "$outfile" ]]; then
    emit_text >"$outfile"
    echo "结果已写入: $outfile"
  else
    emit_text
  fi
fi

exit 0

