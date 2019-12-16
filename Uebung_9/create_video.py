
import os

# os.system("ffmpeg -r 1/5 -i Uebung_9/images/%d.png -c:v libx264 -vf fps=60 -pix_fmt yuv420p out.mp4")
fps = 60
resolution = "1920x1080"
selection_pattern = "images/%d.png"
quality = 5  # lower is better
output_name = "out.mp4"
os.system(f"ffmpeg -r {fps} -s {resolution} -i {selection_pattern} -vcodec libx264 -crf {quality} -pix_fmt yuv420p {output_name}")