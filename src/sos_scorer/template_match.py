from argparse import ArgumentParser
from pathlib import Path

import cv2 as cv
import imutils
import numpy as np
import pandas as pd
from rich import print
from rich.progress import track
from scipy.interpolate import interp1d

from color_consts import skull_colors
from scorer import Scorer
from skull import Skull
import matplotlib.pyplot as plt

TEMPLATE_FOLDER_PATH = r"resources\templates"
THRESHOLD = 0.6
MATCH_METHOD = cv.TM_CCOEFF_NORMED

def draw_text(img, text,
          font=cv.FONT_HERSHEY_SIMPLEX,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):
    """Draw text over a background
    Taken from https://stackoverflow.com/questions/60674501/how-to-make-black-background-in-cv2-puttext-with-python-opencv"""

    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.rectangle(img, pos, (x + text_w, int(y + (text_h * 1.5))), text_color_bg, -1)
    cv.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness, cv.LINE_4)

    return text_size

def auto_digitize(x, y, reverse=False):
    y = np.array(y)

    # Approximately map to linear range
    if reverse:
        y_range = [y.max(),y.min()]
    else:
        y_range = [y.min(),y.max()]

    lin_map = interp1d([x.min(), x.max()], y_range) # Create the map

    mapped_x = np.array(list(map(lin_map, x)))      # Apply the map to the x positions


    # Find the nearest value in the target
    diffs = (mapped_x.reshape(1,-1) - y.reshape(-1,1))  # Difference between mapped values and actual locations
    idxs = np.abs(diffs).argmin(axis=0)                 # Find the minimum difference 

    # Convert to the closest digitzed point
    digitizer = lambda x : y[x]
    digitized_x = np.array(list(map(digitizer, idxs)))

    return digitized_x

def read_templates(fpath : str):
    cwd = Path(__file__).parent.resolve()
    template_folder = cwd.joinpath(fpath) 
    template_pths = list(template_folder.glob("*.png")) # Paths to the template images


    templates = {img_pth.stem : cv.imread(str(img_pth), 0) for img_pth in template_pths}

    return templates

def resize_img(img, scale):
    resized = imutils.resize(img, width = int(img.shape[1] * scale))

    return resized

def init_stack(final_matches):
    stack = {}
    for _, row in final_matches.iterrows():
        stack_x = row['mapped_x']
        stack_y = row['mapped_y']
        skull = row['skull']

        try:
            stack[stack_y]
        except KeyError:
            stack[stack_y] = {}

        stack[stack_y][stack_x] = Skull[skull.upper()]

    return stack

def id_templates(img_gray : np.ndarray, templates):
    matches = []

    scales = np.linspace(1.0, 0.2, 20)  # Generate scales of original image for multi-scale template matching
    
    for scale in track(scales, description='Identifying skulls in image'):
        
        # Resize the image
        resized = resize_img(img_gray, scale)
        r = img_gray.shape[1] / float(resized.shape[1])

        for skull_name, template in templates.items():

            h, w = template.shape

            # Skip iteration if the source image is smaller than the template
            if resized.shape[0] < h or resized.shape[1] < w:
                continue

            # * Matching takes place here
            res = cv.matchTemplate(resized, template, MATCH_METHOD)

            loc = np.where(res >= THRESHOLD)    # Select only the positions which meet the required threshold

            # Save the data for each potential match
            # We reverse the loc to get data in an x, y format, rather than y, x
            for (x, y), match_score in zip(zip(*loc[::-1]), res[loc]):

                # Scale the point to the size of the original image
                x0 = int(x / scale)
                y0 = int(y / scale)

                # Opposite corner of match point
                x1 = x0 + int(w/scale)
                y1 = y0 + int(h/scale)

                match = {
                    'x0' : x0,
                    'y0' : y0,
                    'x1' : x1,
                    'y1' : y1,
                    'skull' : skull_name,
                    'score' : match_score
                }

                matches.append(match)

    res = pd.DataFrame(matches)    

    return res

def create_match_image(img_rgb, final_matches, matches_df):
    match_img = img_rgb.copy()
    
    # Display the selected matches in the original image
    for _, row in final_matches.iterrows():
        stack_x = row['mapped_x']
        stack_y = row['mapped_y']
        skull_name = row['skull']

        # Get the positions of a random correct match
        rand_row = matches_df[(matches_df['mapped_x'] == stack_x) & (matches_df['mapped_y'] == stack_y) & (matches_df['skull'] == skull_name)].iloc[0, :]
        
        color = skull_colors[skull_name]

        # Draw recangle around shield
        cv.rectangle(match_img, (rand_row['x0'], rand_row['y0']), (rand_row['x1'], rand_row['y1']), (0,0,0), 12)
        cv.rectangle(match_img, (rand_row['x0'], rand_row['y0']), (rand_row['x1'], rand_row['y1']), color, 3)     

        draw_text(match_img, skull_name, pos=(rand_row['x1'], rand_row['y1']), text_color=color, font_scale=4, font_thickness=4)
    
    return match_img

def main():
    parser = ArgumentParser(description = "Score a Skulls of Sedlec stack with a picture.")
    parser.add_argument(
        '-f','--fpath',
        help='Filepath of image to process.',
        required=True,
        type=Path
        )
    args = parser.parse_args()

    # Read in image data
    templates = read_templates(TEMPLATE_FOLDER_PATH)
    img_rgb = cv.imread(str(args.fpath))

    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)    # Convert image to gray scale    

    matches_df = id_templates(img_gray, templates)

    tmp_img = img_rgb.copy()

    for _, row in matches_df.iterrows():
        cv.rectangle(tmp_img, (row['x0'], row['y0']), (row['x1'], row['y1']), skull_colors[row['skull']], 3) 

    plt.imshow(cv.cvtColor(tmp_img, cv.COLOR_BGR2RGB))
    plt.title("Results of matching")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    # Map the positions to the 'stack positions'
    stack_xs = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])    # Possible x positions in the stack
    stack_ys = np.array([0, 0.5, 1, 1.5, 2, 2.5])       # Possible y positions in the stack

    matches_df['mapped_x'] = auto_digitize(matches_df['x0'], stack_xs)
    matches_df['mapped_y'] = auto_digitize(matches_df['y0'], stack_ys, reverse=True)

    # Calculate the total sum of each points
    loc_totals = matches_df.groupby(by=['mapped_x', 'mapped_y', 'skull'], axis = 0)['score'].sum()

    # Find the best skull for each position
    final_matches = loc_totals.loc[loc_totals.groupby(by=['mapped_x', 'mapped_y']).idxmax()]
    final_matches = final_matches.reset_index().drop('score', axis = 1)

    match_img = create_match_image(img_rgb, final_matches, matches_df)
    
    # Build final stack
    stack = init_stack(final_matches)

    # Do the calculation and display the result
    scr = Scorer(stack)
    scr.calc_score()
    scr.print_scores()


    # Show the final image with the matched area
    plt.imshow(cv.cvtColor(match_img, cv.COLOR_BGR2RGB))
    plt.title("Results of matching")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()