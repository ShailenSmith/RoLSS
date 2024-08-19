SHORT=a:,d:,c:,h
LONG=arch:,dataset:,cuda:,help
OPTS=$(getopt -a -n train_skip --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
while [ : ]; do
  case "$1" in
    -a | --arch)
        ARCH=$2
        shift 2
        ;;
    -d | --dataset)
        DATA=$2
        shift 2
        ;;
    -c | --cuda)
        CUDA=$2
        shift 2
        ;;
    -h | --help)
        echo "Specify argument and values in the format shown below:
        [-a | --arch] : (str) [random | uniform | small]
        [-d | --dataset] : (str) [CelebA | FaceScrub | Stanford_Dogs]
        [-w | --weight] : (str) [Path to weight file (.pth)]"
        shift
        ;;
    --) shift;
        break
        ;;
  esac
done

CONFIG=configs/training/targets/${DATA}/${ARCH}/skip.yaml

CUDA_VISIBLE_DEVICES=$CUDA python train_target.py -c $CONFIG