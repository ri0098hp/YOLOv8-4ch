function usage {
  cat <<EOM
Usage: $(basename "$0") [OPTION]...
  -h Display help
  -a attached container
  -b buid from Dockerfile
  -d debug (you should build image manually first)
  -e exec into container
  -r simple run
  -s down the container
EOM
  exit 2
}


while getopts "abcdersh" optKey; do
  case "$optKey" in
    a)
      docker restart YOLOv8-okuda
      docker attach YOLOv8-okuda
      ;;
    b)
      docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t yolov8-okuda:latest -f ./docker/Dockerfile .
      ;;
    c)
      docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t yolov8-okuda:test -f ./docker/Dockerfile-test .
      ;;
    d)
      docker run --name YOLOv8-okuda-debug --gpus all -it --shm-size=16g --ipc=host --rm \
      --mount type=bind,source="$(pwd)",target=/usr/src/ultralytics/ \
      yolov8-okuda:test
      ;;
    r)
      docker run --name YOLOv8-okuda --gpus all -it --shm-size=16g --ipc=host \
      --mount type=bind,source="$(pwd)"/data,target=/usr/src/ultralytics/data \
      --mount type=bind,source="$(pwd)"/datasets,target=/usr/src/ultralytics/datasets \
      --mount type=bind,source="$(pwd)"/runs,target=/usr/src/ultralytics/runs \
      yolov8-okuda
      ;;
    s)
      docker rm YOLOv8-okuda
      ;;
    '-h'|'--help'|* )
      usage
      ;;
  esac
done
