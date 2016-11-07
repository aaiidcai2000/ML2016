import sys
from PIL import Image

im = Image.open(sys.argv[1])
imR = im.transpose(Image.ROTATE_180)
imR.save("ans2.png")
