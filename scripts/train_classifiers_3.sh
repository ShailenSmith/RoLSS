################################### INSTRUCTIONS ###################################

# bash scripts/train/full.sh -a <ARCHITECTURE> -d <DATASET> -c <CUDA-DEVICE>
# bash scripts/train/skip.sh -a <ARCHITECTURE> -d <DATASET> -c <CUDA-DEVICE>

# FOR MORE INFORMATION, PLEASE REFER TO THE BASH SCRIPT FILES IN "scripts/train" FOLDER.

# EXAMPLE:
# bash scripts/train/full.sh -a resnet152 -d Stanford_Dogs -c 0
# bash scripts/train/skip.sh -a resnet152 -d Stanford_Dogs -c 0

################################### END INSTRUCTIONS ###################################

# bash scripts/train/full.sh -a vit -d FaceScrub -c 0
# bash scripts/train/skip.sh -a vit -d FaceScrub -c 1

# bash scripts/train/skip.sh -a resnet50 -d FaceScrub -c 3

# bash scripts/train/skip.sh -a densenet169 -d FaceScrub -c 3

bash scripts/train/skip.sh -a maxvit -d FaceScrub -c 3

# bash scripts/train/skip.sh -a resnet152 -d FaceScrub -c 3
