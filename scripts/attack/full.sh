SHORT=a:,c:,d:,w:,h
LONG=arch:,cuda:,dataset:,weight:,help
OPTS=$(getopt -a -n attack_skip --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
while [ : ]; do
  case "$1" in
    -a | --arch)
        ARCH=$2
        shift 2
        ;;
    -c | --cuda)
        CUDA=$2
        shift 2
        ;;
    -d | --dataset)
        DATA=$2
        shift 2
        ;;
    -w | --weight)
        WEIGHT=$2
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

LOG_FOLDER="logs"

# TRAIN_LOG="logs/deeplabv3plus_alphanet_${ARCH}_train.log"
# TEST_LOG="logs/deeplabv3plus_alphanet_${ARCH}_test.log"
# LATEST_CKPT="checkpoints/latest_deeplabv3plus_alphanet_${ARCH}_voc_os16.pth"
# BEST_CKPT="checkpoints/best_deeplabv3plus_alphanet_${ARCH}_voc_os16.pth"

# SAVE_FOLDER="experiments/alphanet_IN100/${METHOD}_sampling/MBV${BLOCK}/deeplabv3plus_alphanet_${ARCH}_MBV${BLOCK}"

mkdir $LOG_FOLDER
# mkdir -p $SAVE_FOLDER

# mv -n $TRAIN_LOG $TEST_LOG $SAVE_FOLDER
# mv -n $BEST_CKPT $LATEST_CKPT $SAVE_FOLDER


WEIGHT_LOG=${WEIGHT##*/}

prefix="Classifier_"
suffix=".pth"
WEIGHT_LOG=${WEIGHT_LOG#"$prefix"}
WEIGHT_LOG=${WEIGHT_LOG%"$suffix"}

CONFIG=configs/attacking/${DATA}/${ARCH}/full.yaml
LOG="logs/${ARCH}_Full_${WEIGHT_LOG}"
IMAGE_LOG="images/${ARCH}_Full_${WEIGHT_LOG}"

CUDA_VISIBLE_DEVICES=$CUDA python attack.py -c $CONFIG -l $LOG -w $WEIGHT -i $IMAGE_LOG