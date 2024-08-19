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

# bash scripts/attack/full.sh -a densenet161 -c "0,1,2,3" -d FaceScrub -w "results/FaceScrub/densenet161_FULL_Scratch/Classifier_0.9393_Epoch_99.pth"

# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.2]" -w "results/FaceScrub/densenet121_skip_4_0.2_scale/Classifier_0.9476_Epoch_99.pth"
# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.4]" -w "results/FaceScrub/densenet121_skip_4_0.4_scale/Classifier_0.9486_Epoch_99.pth"
# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.6]" -w "results/FaceScrub/densenet121_skip_4_0.6_scale/Classifier_0.9481_Epoch_99.pth"
# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.8]" -w "results/FaceScrub/densenet121_skip_4_0.8_scale/Classifier_0.9481_Epoch_99.pth"

# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.2]" -w "results/FaceScrub/densenet121_skip_4_0.2_scale/Classifier_0.9203_Epoch_70.pth"
# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.4]" -w "results/FaceScrub/densenet121_skip_4_0.4_scale/Classifier_0.9217_Epoch_67.pth"
# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.6]" -w "results/FaceScrub/densenet121_skip_4_0.6_scale/Classifier_0.9203_Epoch_68.pth"
# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.8]" -w "results/FaceScrub/densenet121_skip_4_0.8_scale/Classifier_0.9196_Epoch_59.pth"

# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.2]" -w "results/FaceScrub/densenet121_skip_4_0.2_scale/Classifier_0.9017_Epoch_40.pth"
# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.4]" -w "results/FaceScrub/densenet121_skip_4_0.4_scale/Classifier_0.8962_Epoch_28.pth"
# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.6]" -w "results/FaceScrub/densenet121_skip_4_0.6_scale/Classifier_0.8978_Epoch_37.pth"
# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.8]" -w "results/FaceScrub/densenet121_skip_4_0.8_scale/Classifier_0.9038_Epoch_35.pth"

# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.2]" -w "results/FaceScrub/densenet121_skip_4_0.2_scale/Classifier_0.8691_Epoch_21.pth"
# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.4]" -w "results/FaceScrub/densenet121_skip_4_0.4_scale/Classifier_0.8737_Epoch_22.pth"
# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.6]" -w "results/FaceScrub/densenet121_skip_4_0.6_scale/Classifier_0.8776_Epoch_17.pth"
# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.8]" -w "results/FaceScrub/densenet121_skip_4_0.8_scale/Classifier_0.8742_Epoch_18.pth"

# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.2]" -w "results/FaceScrub/densenet121_skip_4_0.2_scale/Classifier_0.8014_Epoch_10.pth"
# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.4]" -w "results/FaceScrub/densenet121_skip_4_0.4_scale/Classifier_0.8039_Epoch_10.pth"
# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.6]" -w "results/FaceScrub/densenet121_skip_4_0.6_scale/Classifier_0.8083_Epoch_9.pth"
# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.8]" -w "results/FaceScrub/densenet121_skip_4_0.8_scale/Classifier_0.8139_Epoch_10.pth"

# bash scripts/attack/skip.sh -a densenet121 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.01]" -w "results/FaceScrub/densenet121_skip_4_0.01_scratch/Classifier_0.9173_Epoch_99.pth"
# bash scripts/attack/skip.sh -a densenet161 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.01]" -w "results/FaceScrub/densenet161_skip_4_0.01_scratch/Classifier_0.9124_Epoch_72.pth"
# bash scripts/attack/skip.sh -a densenet161 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.01]" -w "results/FaceScrub/densenet161_skip_4_0.01_scratch/Classifier_0.9319_Epoch_75.pth"
# bash scripts/attack/skip.sh -a densenet169 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.01]" -w "results/FaceScrub/densenet169_skip_4_0.01_scratch/Classifier_0.9295_Epoch_99.pth"
# bash scripts/attack/skip.sh -a densenet201 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.01]" -w "results/FaceScrub/densenet201_skip_4_0.01_scratch/Classifier_0.9309_Epoch_99.pth"

# bash scripts/attack/skip.sh -a densenet161 -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -w "results/FaceScrub/densenet161_skip_1/Classifier_0.9194_Epoch_99.pth"
# bash scripts/attack/skip.sh -a densenet161 -c "0,1,2,3" -d FaceScrub -s "[1,0,1,1]" -w "results/FaceScrub/densenet161_skip_2/Classifier_0.9219_Epoch_99.pth"

# bash scripts/attack/skip.sh -a densenet201 -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -w "results/FaceScrub/densenet201_skip_1/Classifier_0.9247_Epoch_99.pth"
# bash scripts/attack/skip.sh -a densenet201 -c "0,1,2,3" -d FaceScrub -s "[1,0,1,1]" -w "results/FaceScrub/densenet201_skip_2/Classifier_0.9048_Epoch_99.pth"

# bash scripts/attack/skip.sh -a resnet50 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.2]" -w "results/FaceScrub/resnet50_skip_4_0.2/Classifier_0.9305_Epoch_99.pth"

# bash scripts/attack/skip.sh -a resnet34 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.2]" -w "results/FaceScrub/resnet34_skip_4_0.2/Classifier_0.9421_Epoch_99.pth"

# bash scripts/attack/full.sh -a resnet101 -c "0,1,2,3" -d FaceScrub -w "results/FaceScrub/resnet101_FULL/Classifier_0.9421_Epoch_99.pth"
# bash scripts/attack/skip.sh -a resnet101 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0]" -w "results/FaceScrub/resnet101_skip_4/Classifier_0.9240_Epoch_99.pth"
# bash scripts/attack/skip.sh -a resnet101 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0.2]" -w "results/FaceScrub/resnet101_skip_4_0.2/Classifier_0.9363_Epoch_99.pth"

# bash scripts/attack/skip.sh -a vit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1,1,1,1,1,1,1,1,1]" -m "attention" -w "results/FaceScrub/vit_skip_1_attention/Classifier_0.6795_Epoch_99.pth"
# bash scripts/attack/skip.sh -a vit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1,1,1,1,1,1,1,1,1]" -m "attention" -w "results/FaceScrub/vit_skip_1_attention/Classifier_0.6554_Epoch_75.pth"

# bash scripts/attack/skip.sh -a vit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1,1,1,1,1,1,1,1,1]" -m "mlp" -w "results/FaceScrub/vit_skip_1_mlp/Classifier_0.7615_Epoch_99.pth"
# bash scripts/attack/skip.sh -a vit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1,1,1,1,1,1,1,1,1]" -m "mlp" -w "results/FaceScrub/vit_skip_1_mlp/Classifier_0.6818_Epoch_42.pth"
# bash scripts/attack/skip.sh -a vit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1,1,1,1,1,1,1,1,1]" -m "mlp" -w "results/FaceScrub/vit_skip_1_mlp/Classifier_0.6535_Epoch_31.pth"

# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "mlp" -w "results/FaceScrub/maxvit_skip_1_mlp/Classifier_0.9506_Epoch_99.pth"
# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "mlp" -w "results/FaceScrub/maxvit_skip_1_mlp/Classifier_0.9409_Epoch_75.pth"
# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "mlp" -w "results/FaceScrub/maxvit_skip_1_mlp/Classifier_0.9217_Epoch_57.pth"
# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "mlp" -w "results/FaceScrub/maxvit_skip_1_mlp/Classifier_0.8959_Epoch_19.pth"

# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "mlp" -w "results/FaceScrub/maxvit_skip_1_mlp_*/Classifier_0.9458_Epoch_99.pth"
# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "mlp" -w "results/FaceScrub/maxvit_skip_1_mlp_*/Classifier_0.9231_Epoch_70.pth"
# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "mlp" -w "results/FaceScrub/maxvit_skip_1_mlp_*/Classifier_0.9003_Epoch_31.pth"

# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "attention" -w "results/FaceScrub/maxvit_skip_1_attention/Classifier_0.9402_Epoch_99.pth"
# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "attention" -w "results/FaceScrub/maxvit_skip_1_attention/Classifier_0.9214_Epoch_73.pth"
# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "attention" -w "results/FaceScrub/maxvit_skip_1_attention/Classifier_0.8994_Epoch_36.pth"

# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "attention" -w "results/FaceScrub/maxvit_skip_1_attention_*/Classifier_0.9379_Epoch_99.pth"
# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "attention" -w "results/FaceScrub/maxvit_skip_1_attention_*/Classifier_0.9226_Epoch_75.pth"
# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "attention" -w "results/FaceScrub/maxvit_skip_1_attention_*/Classifier_0.9022_Epoch_46.pth"

# bash scripts/attack/skip.sh -a resnet101 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0]" -m "all" -w "results/FaceScrub/resnet101_skip_3(0.8)&4/Classifier_0.8871_Epoch_99.pth"
# bash scripts/attack/skip.sh -a resnet50 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0]" -m "all" -w "results/FaceScrub/resnet50_skip_3(0.9)&4/Classifier_0.9251_Epoch_99.pth"
# bash scripts/attack/skip.sh -a resnet34 -c "0,1,2,3" -d FaceScrub -s "[1,1,1,0]" -m "all" -w "results/FaceScrub/resnet34_skip_3(0.9)&4/Classifier_0.9349_Epoch_99.pth"

# bash scripts/attack/full.sh -a mobilenetv2 -c "0,1,2,3" -d FaceScrub -w "results/FaceScrub/mobilenetv2_FULL/Classifier_0.9597_Epoch_99.pth"

# bash scripts/attack/full.sh -a deit -c "0,1,2,3" -d FaceScrub -w "results/FaceScrub/deit_FULL/Classifier_0.6079_Epoch_22.pth"
# bash scripts/attack/full.sh -a deit -c "0,1,2,3" -d FaceScrub -w "results/FaceScrub/deit_FULL/Classifier_0.6501_Epoch_34.pth"
# bash scripts/attack/full.sh -a deit -c "0,1,2,3" -d FaceScrub -w "results/FaceScrub/deit_FULL/Classifier_0.7092_Epoch_76.pth"
bash scripts/attack/full.sh -a deit -c "0,1,2,3" -d FaceScrub -w "results/FaceScrub/deit_FULL/Classifier_0.7203_Epoch_99.pth"

# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "mlp" -w "results/FaceScrub/maxvit_skip_1.1_block_mlp/Classifier_0.8934_Epoch_4.pth"
# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "mlp" -w "results/FaceScrub/maxvit_skip_1.1_block_mlp/Classifier_0.9138_Epoch_12.pth"
# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "mlp" -w "results/FaceScrub/maxvit_skip_1.1_block_mlp/Classifier_0.9666_Epoch_99.pth"

# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,0,1,1]" -m "mlp" -w "results/FaceScrub/maxvit_skip_1&2_block_mlp/Classifier_0.8987_Epoch_55.pth"
# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,0,1,1]" -m "mlp" -w "results/FaceScrub/maxvit_skip_1&2_block_mlp/Classifier_0.9136_Epoch_70.pth"
# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,0,1,1]" -m "mlp" -w "results/FaceScrub/maxvit_skip_1&2_block_mlp/Classifier_0.9395_Epoch_99.pth"

# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "attention" -w "results/FaceScrub/maxvit_skip_1.1_block_attention/Classifier_0.8969_Epoch_24.pth"
# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "attention" -w "results/FaceScrub/maxvit_skip_1.1_block_attention/Classifier_0.9156_Epoch_52.pth"
# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,1,1,1]" -m "attention" -w "results/FaceScrub/maxvit_skip_1.1_block_attention/Classifier_0.9513_Epoch_99.pth"

# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,0,1,1]" -m "attention" -w "results/FaceScrub/maxvit_skip_1&2_block_attention/Classifier_0.8976_Epoch_49.pth"
# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,0,1,1]" -m "attention" -w "results/FaceScrub/maxvit_skip_1&2_block_attention/Classifier_0.9110_Epoch_71.pth"
# bash scripts/attack/skip.sh -a maxvit -c "0,1,2,3" -d FaceScrub -s "[0,0,1,1]" -m "attention" -w "results/FaceScrub/maxvit_skip_1&2_block_attention/Classifier_0.9379_Epoch_99.pth"

# bash scripts/attack/full.sh -a llava -c "0,1,2,3" -d FaceScrub