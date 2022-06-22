import sys
import os
import glob
from PIL import Image


def main():
    img_dir = "wukong"
    jpg_ext = ".jpg"
    angles = [90, 180, 270]
    for file_name in glob.iglob(os.path.join(img_dir, "*" + jpg_ext)):
        image = Image.open(file_name)
        for angle in angles:
            rot_suffix = "_r{:03d}{:s}".format(angle, jpg_ext)
            file_name_rot = file_name.replace(jpg_ext, rot_suffix)
            image_rot = image.rotate(angle)
            image_rot.save(file_name_rot)
            print("Rotated: {:s} by {:3d} degrees to {:s}".format(
                file_name, angle, file_name_rot))


if __name__ == "__main__":
    print("Python {:s} on {:s}\n".format(sys.version, sys.platform))
    main()
    print("\nDone.")
