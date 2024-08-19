import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

### DEFINE VARIABLES ###
full_images_path = "viz/maxvit_FULL"
skip_images_path = "viz/maxvit_skip_1110"
real_images_path = "viz/real images"

root_save_path = "mturk_data"
url_prefix = "https://ppa-mi-skip-connection.s3.ap-southeast-1.amazonaws.com"
### END VARIABLES ###

num_IDs = 530

def create_collage(id, image_a_path, image_b_path, real_image_path, save_path):
    ### EDIT VARIABLES HERE ###
    h_margin = 50
    v_margin = 50
    text_height = 80
    label_real = "Choose the photo below that best\nrepresents the target identity above"
    label_a = "A"
    label_b = "B"
    font = cv2.FONT_HERSHEY_SIMPLEX
    ### END VARIABLES ###

    image_a = cv2.cvtColor(cv2.imread(image_a_path), cv2.COLOR_BGR2RGB)
    image_b = cv2.cvtColor(cv2.imread(image_b_path), cv2.COLOR_BGR2RGB)
    real_image = cv2.cvtColor(cv2.imread(real_image_path), cv2.COLOR_BGR2RGB)

    # Ensure that real image is proportional to other images
    max_dimension = 224

    # Get the original dimensions of the image
    height, width, _ = real_image.shape

    # Calculate the new dimensions while maintaining the aspect ratio
    if width > height:
        new_width = max_dimension
        new_height = int((max_dimension / width) * height)
    else:
        new_height = max_dimension
        new_width = int((max_dimension / height) * width)

    # Resize the image
    real_image = cv2.resize(real_image, (new_width, new_height))

    canvas_height = image_a.shape[1] + text_height + v_margin + real_image.shape[1] + text_height + 10
    canvas_width = image_a.shape[0] + h_margin + image_b.shape[0] 

    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # Fill with white (255) background

    # Real Image
    real_x_start = int((canvas_width - real_image.shape[1]) / 2)
    real_x_end = real_x_start + real_image.shape[1]
    real_y_start = 0
    real_y_end = real_y_start + real_image.shape[0]

    canvas[real_y_start:real_y_end, real_x_start:real_x_end] = real_image

    # Real Label
    label_real_image = np.ones((text_height, canvas_width, 3), dtype=np.uint8) * 255  # Create a blank image for the label
    dy = 10
    for i, line in enumerate(label_real.split('\n')):
        textsize = cv2.getTextSize(line, font, 0.8, 2)[0]
        textX = int((canvas_width - textsize[0]) / 2)
        textY = (i+1) * textsize[1] + i*dy + 10
        cv2.putText(label_real_image, line, (textX, textY), font, 0.8, (0, 0, 0), 2)  # Add label text
    canvas[real_y_end+v_margin:real_y_end+v_margin+text_height, :] = label_real_image

    # Image A
    image_a_x_start = 0
    image_a_x_end = image_a_x_start + image_a.shape[1]
    image_a_y_start = real_y_end + label_real_image.shape[0] + v_margin
    image_a_y_end = image_a_y_start + image_a.shape[0]
    canvas[image_a_y_start:image_a_y_end, image_a_x_start:image_a_x_end] = image_a

    # Label A
    textsize = cv2.getTextSize(label_a, font, 2, 2)[0]
    textX = int((image_a.shape[1] - textsize[0]) / 2)
    textY = textsize[1] + 10
    label_image_a = np.ones((text_height, image_a.shape[1], 3), dtype=np.uint8) * 255  # Create a blank image for the label
    cv2.putText(label_image_a, label_a, (textX, textY), font, 2, (0, 0, 0), 2)  # Add label text
    canvas[image_a_y_end:image_a_y_end+text_height, image_a_x_start:image_a_x_end] = label_image_a

    # Image B
    image_b_x_start = image_a_x_end + h_margin
    image_b_x_end = image_b_x_start + image_b.shape[1]
    image_b_y_start = real_y_end + label_real_image.shape[0] + v_margin
    image_b_y_end = image_b_y_start + image_b.shape[0]
    canvas[image_b_y_start:image_b_y_end, image_b_x_start:image_b_x_end] = image_b
    
    # Label B
    textsize = cv2.getTextSize(label_b, font, 2, 2)[0]
    textX = int((image_b.shape[1] - textsize[0]) / 2)
    textY = textsize[1] + 10
    label_image_b = np.ones((text_height, image_b.shape[1], 3), dtype=np.uint8) * 255  # Create a blank image for the label
    cv2.putText(label_image_b, label_b, (textX, textY), font, 2, (0, 0, 0), 2)  # Add label text
    canvas[image_b_y_end:image_b_y_end+text_height, image_b_x_start:image_b_x_end] = label_image_b

    plt.axis('off')
    plt.imshow(canvas)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return canvas


def main():
    df_gt_data = pd.DataFrame(columns=["item1_image_url", "item2_image_url", "item3_image_url", "A", "B"])
    df_aws_submit = pd.DataFrame(columns=["item1_image_url"])

    os.makedirs(root_save_path, exist_ok=True)

    for id in range(num_IDs):

        # Select image from full model
        full_path = os.path.join(full_images_path, f"ID_{id}")
        full_images = os.listdir(full_path)
        full_image = os.path.join(full_path, random.choice(full_images))

        # Select image from skip-n removed model
        skip_path = os.path.join(skip_images_path, f"ID_{id}")
        skip_images = os.listdir(skip_path)
        skip_image = os.path.join(skip_path, random.choice(skip_images))

        # Select image from target identity
        real_path = os.path.join(real_images_path, f"ID_{id}")
        real_images = os.listdir(real_path)
        real_image = os.path.join(real_path, random.choice(real_images))

        save_path = os.path.join(root_save_path, f"ID_{id}")

        # Shuffle orientation of option A and B
        choices = [0,1]
        choice = random.choice(choices)
        item1_url, item2_url, item3_url = None, None, None
        a, b, my_answer = None, None, None

        if choice:
            item1_url = real_image
            item2_url = full_image
            item3_url = skip_image
            a = 'full'
            b = 'skip_4 removed'
            my_answer = 'A'
        
        else:
            item1_url = real_image
            item2_url = skip_image
            item3_url = full_image
            a = 'skip_4 removed'
            b = 'full'
            my_answer = 'B'

        row_gt_data = {"item1_image_url": item1_url, 
                       "item2_image_url": item2_url, 
                       "item3_image_url": item3_url,
                       "ID": id,
                       'A': a,
                       'B': b,
                       'my_answer': my_answer}
        df_gt_data_tmp = pd.DataFrame(row_gt_data, index=[0])
        df_gt_data = pd.concat([df_gt_data,df_gt_data_tmp], ignore_index=True)

        create_collage(id=id,
                       image_a_path=item2_url, 
                       image_b_path=item3_url, 
                       real_image_path=item1_url, 
                       save_path=save_path)
        
        collage_url = url_prefix + "/" + save_path + ".png"

        row_aws_data = {"item1_image_url":collage_url, 
                        # "item2_image_url": save_path,
                        "ID": id,
                        "A": a,
                        "B": b, 
                        "my_answer": my_answer
                    }
        df_aws_submit_tmp = pd.DataFrame(row_aws_data, index=[0])
        df_aws_submit = pd.concat([df_aws_submit,df_aws_submit_tmp], ignore_index=True)


    df_gt_data["ID"] = df_gt_data["ID"].astype("Int16")
    df_aws_submit["ID"] = df_aws_submit["ID"].astype("Int16")

    df_gt_data.to_csv(f"{root_save_path}/gt.csv", index=False)
    df_aws_submit.to_csv(f"{root_save_path}/aws_submit.csv", index=False)


if __name__ == '__main__':
    main()