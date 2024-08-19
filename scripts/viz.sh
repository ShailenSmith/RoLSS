################################### INSTRUCTIONS ###################################

# 1. Run the following code with the specified parameters:

#    -c : <PATH/TO/CONFIG/FILE>
#    -w : <PATH/TO/LATENT/FILE>
#    -s : <SKIP_CONFIGURATION> (Optional)

# EXAMPLE:

# python viz.py -c configs/attacking/FaceScrub/maxvit/skip.yaml -s 1110 -w results/optimized_w_selected_3zj76wa1.pt 

################################### END INSTRUCTIONS ###################################

python viz.py -c configs/attacking/FaceScrub/resnet101/full.yaml -w "images/resnet101_Full_0.9421_Epoch_99/optimized_w_selected_u0pw9t16.pt" 
python viz.py -c configs/attacking/FaceScrub/resnet101/skip.yaml -s "[1,1,1,0]" -w "images/resnet101_Skip_[1,1,1,0]_0.9240_Epoch_99/optimized_w_selected_kf6q9hgg.pt" 
python viz.py -c configs/attacking/FaceScrub/resnet101/skip.yaml -s "[1,1,1,0.2]" -w "images/resnet101_Skip_[1,1,1,0.2]_0.9363_Epoch_99/optimized_w_selected_c5k445kd.pt"