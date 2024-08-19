################################### INSTRUCTIONS ###################################

# bash scripts/train/full.sh -a <ARCHITECTURE> -d <DATASET> -c <CUDA-DEVICE> -w <PATH/TO/WEIGHT/FILE>
# bash scripts/train/skip.sh -a <ARCHITECTURE> -d <DATASET> -c <CUDA-DEVICE> -w <PATH/TO/WEIGHT/FILE> -s <SKIP-CONFIGURATION> 

# FOR MORE INFORMATION, PLEASE REFER TO THE BASH SCRIPT FILES IN "scripts/attack" FOLDER.

# EXAMPLE:
# bash scripts/attack/full.sh -a densenet169 -c "0,1,2,3" -d Stanford_Dogs -w "results/Stanford_Dogs/densenet169_FULL/Classifier_0.5700_Epoch_99.pth"
# bash scripts/attack/skip.sh -a densenet169 -c "0,1,2,3" -d Stanford_Dogs -s 1110 -w "results/Stanford_Dogs/densenet169_skip_4/Classifier_0.3980_Epoch_99.pth"

################################### END INSTRUCTIONS ###################################

# bash scripts/attack/skip.sh -a resnet101 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.8]" -w "results/FaceScrub/resnet101_skip_4_0.8/Classifier_0.9481_Epoch_99.pth"
# bash scripts/attack/skip.sh -a resnet101 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.8]" -w "results/FaceScrub/resnet101_skip_4_0.8/Classifier_0.9080_Epoch_56.pth"
# bash scripts/attack/skip.sh -a resnet101 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.8]" -w "results/FaceScrub/resnet101_skip_4_0.8/Classifier_0.8542_Epoch_14.pth"

# bash scripts/attack/skip.sh -a resnet101 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.6]" -w "results/FaceScrub/resnet101_skip_4_0.6/Classifier_0.9455_Epoch_99.pth"
# bash scripts/attack/skip.sh -a resnet101 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.6]" -w "results/FaceScrub/resnet101_skip_4_0.6/Classifier_0.9078_Epoch_68.pth"
# bash scripts/attack/skip.sh -a resnet101 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.6]" -w "results/FaceScrub/resnet101_skip_4_0.6/Classifier_0.8563_Epoch_21.pth"

# bash scripts/attack/skip.sh -a resnet101 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.4]" -w "results/FaceScrub/resnet101_skip_4_0.4/Classifier_0.9395_Epoch_99.pth"
# bash scripts/attack/skip.sh -a resnet101 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.4]" -w "results/FaceScrub/resnet101_skip_4_0.4/Classifier_0.9052_Epoch_72.pth"
# bash scripts/attack/skip.sh -a resnet101 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.4]" -w "results/FaceScrub/resnet101_skip_4_0.4/Classifier_0.8575_Epoch_29.pth"

# bash scripts/attack/skip.sh -a resnet101 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.2]" -w "results/FaceScrub/resnet101_skip_4_0.2/Classifier_0.9379_Epoch_99.pth"
# bash scripts/attack/skip.sh -a resnet101 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.2]" -w "results/FaceScrub/resnet101_skip_4_0.2/Classifier_0.9008_Epoch_63.pth"
# bash scripts/attack/skip.sh -a resnet101 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.2]" -w "results/FaceScrub/resnet101_skip_4_0.2/Classifier_0.8568_Epoch_36.pth"

# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.5]" -w "results/FaceScrub/densenet121_skip_4_0.5/Classifier_0.9618_Epoch_99.pth"
# bash scripts/attack/skip.sh -a densenet161 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.5]" -w "results/FaceScrub/densenet161_skip_4_0.5/Classifier_0.9499_Epoch_99.pth"
# bash scripts/attack/skip.sh -a densenet169 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.5]" -w "results/FaceScrub/densenet169_skip_4_0.5/Classifier_0.9458_Epoch_99.pth"
# bash scripts/attack/skip.sh -a densenet201 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.5]" -w "results/FaceScrub/densenet201_skip_4_0.5/Classifier_0.9330_Epoch_99.pth"

# bash scripts/attack/full.sh -a resnet50 -c "0,1,2,3" -d FaceScrub -w "results/FaceScrub/resnet50_FULL/Classifier_0.9458_Epoch_99.pth"

# bash scripts/attack/full.sh -a densenet201 -c "0,1,2,3" -d FaceScrub -w "results/FaceScrub/densenet201_FULL_Scratch/Classifier_0.9432_Epoch_99.pth"

bash scripts/evaluate/full.sh -a maxvit -c "0,1,2,3" -d FaceScrub -w "results/FaceScrub/densenet161_FULL_Scratch/Classifier_0.9393_Epoch_99.pth"
