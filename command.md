```
bash -c 'cmd="$*"; "$@"; echo "+ $cmd"' -- \
docker run  --rm --gpus "device=0"  \
  -v /root/smartiot_docker/DeDiff:/workspace   -it \
    -w /workspace     gnn_image    \
        python main.py --dataset twitter 
```

```
./gpumon_docker.sh \
bash -c 'cmd="$*"; "$@"; echo "+ $cmd"' -- \
docker run  --rm --gpus "device=0"  \
  -v /root/smartiot_docker/DeDiff:/workspace   -it \
    -w /workspace     gnn_image    \
        python main.py --dataset twitter 


./gpumon_docker.sh \
bash -c 'cmd="$*"; "$@"; echo "+ $cmd"' -- \
docker run  --rm --gpus "device=1"  \
  -v /root/smartiot_docker/DeDiff:/workspace   -it \
    -w /workspace     gnn_image    \
        python main.py --dataset android 

./gpumon_docker.sh \
bash -c 'cmd="$*"; "$@"; echo "+ $cmd"' -- \
docker run  --rm --gpus "device=2"  \
  -v /root/smartiot_docker/DeDiff:/workspace   -it \
    -w /workspace     gnn_image    \
        python main.py --dataset christian 

./gpumon_docker.sh \
bash -c 'cmd="$*"; "$@"; echo "+ $cmd"' -- \
docker run  --rm --gpus "device=3"  \
  -v /root/smartiot_docker/DeDiff:/workspace   -it \
    -w /workspace     gnn_image    \
        python main.py --dataset douban 
```