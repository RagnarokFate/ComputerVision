import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import AssignmentSolution
import FolderDecoder


def make_sides_black(im_path):
    old_im = Image.open(im_path)
    old_size = old_im.size

    new_size = (800, 800)
    new_im = Image.new("RGB", new_size)
    box = tuple((n - o) // 2 for n, o in zip(new_size, old_size))
    new_im.paste(old_im, box)

    return new_im


def combine_images_color(img1, img2, img3):
    # Find the maximum size for each dimension
    max_width = max(img1.shape[1], img2.shape[1], img3.shape[1])
    max_height = max(img1.shape[0], img2.shape[0], img3.shape[0])

    # Create a new black image with the maximum size
    new_img = np.zeros((max_height, max_width, 3), dtype=np.uint8)

    # Resize each image to fit the maximum size
    resized_img1 = cv2.resize(img1, (max_width, max_height))
    resized_img2 = cv2.resize(img2, (max_width, max_height))
    resized_img3 = cv2.resize(img3, (max_width, max_height))

    # Loop over each pixel location and set the value to the maximum pixel value for that location
    for i in range(max_height):
        for j in range(max_width):
            max_value_r = max(resized_img1[i, j, 0], resized_img2[i, j, 0], resized_img3[i, j, 0])
            max_value_g = max(resized_img1[i, j, 1], resized_img2[i, j, 1], resized_img3[i, j, 1])
            max_value_b = max(resized_img1[i, j, 2], resized_img2[i, j, 2], resized_img3[i, j, 2])
            new_img[i, j] = (max_value_r, max_value_g, max_value_b)

    # Return the new image
    return new_img


# this function gets all images paths and makes all their bounds black
# returns them in a list
def make_all_ims_sides_black(ims_paths_list):
    if len(ims_paths_list) == 0:
        print("make all error list len is 0 !!!!!")
        return 0

    new_list = []
    for i in ims_paths_list:
        old_im = Image.open(i)
        old_size = old_im.size

        new_size = (800, 800)
        new_im = Image.new("RGB", new_size)  ## luckily, this is already black!
        box = tuple((n - o) // 2 for n, o in zip(new_size, old_size))
        new_im.paste(old_im, box)

        cv_im = np.array(new_im)

        cv_im = cv2.cvtColor(cv_im, cv2.COLOR_RGB2BGR)

        new_list.append(cv_im)

    return new_list


# this function finds a match for image in index 0 in the rest of the list and returns everything needed
# input list of images
def find_a_match_in_list(ims_list):
    keyPoints1, descriptors1 = AssignmentSolution.SIFT_transform(cv2.cvtColor(ims_list[0], cv2.COLOR_BGR2GRAY))
    keyPoints_i, descriptors_i = None, None
    matches = None
    index = 1
    for i in ims_list[1:]:

        keyPoints_i, descriptors_i = AssignmentSolution.SIFT_transform(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY))
        matches, is_match = AssignmentSolution.find_matrix_then_matches(descriptors1, descriptors_i)

        if is_match:
            break

        index += 1
    if keyPoints_i == None or matches == None:
        print("error - match wasn't found ")

    return index, keyPoints1, keyPoints_i, matches


def stitch_pair(im1, im2, transformation):
    max_width = max(im1.shape[1], im2.shape[1])
    max_height = max(im1.shape[0], im2.shape[0])
    black_image = np.zeros((max_height, max_width, 3), np.uint8)

    warped_im = cv2.warpPerspective(im1, transformation,
                                    (im2.shape[0], im2.shape[1]))

    im = combine_images_color(im2, warped_im, black_image)

    return im


def main_():
    folders_paths = FolderDecoder.LoadFolders()
    for i in range(0, 10):
        print(folders_paths[i])
        Puzzle_Folder_Path, Type, Textfile_Name, TextFile_Height, Textfile_Width, TextFile_Matrix, puzzle_pieces, Puzzle_Pieces_Number, pieces_paths = FolderDecoder.PuzzleLoader(
            'puzzles/' + folders_paths[i])

        # make sides black
        for j in range (0,2):
            ims_list = make_all_ims_sides_black(pieces_paths)
            # ims_list.reverse()

            while (len(ims_list) != 1):
                # find a match for the first one
                index, keyPoints1, keyPoints_i, matches = find_a_match_in_list(ims_list)

                # find the best transformation for the match that was found
                best_trans = AssignmentSolution.RANSAC_Implementation(matches, keyPoints1, keyPoints_i, Type)

                if Type == "affine":
                    # make it A_homogeneous
                    A_homogeneous = np.zeros((3, 3))
                    A_homogeneous[:2, :] = best_trans
                    A_homogeneous[2, 2] = 1
                    best_trans = A_homogeneous

                stitched_im = stitch_pair(ims_list[0], ims_list[index], best_trans)
                del ims_list[index]
                ims_list[0] = stitched_im

            stitched_image = ims_list[0]
        CROPED_IMAGE = AssignmentSolution.CROP_IMAGE(stitched_image,Textfile_height=TextFile_Height,Textfile_width=Textfile_Width)
        cv2.imwrite('Outputs' + '/' + str(folders_paths[i]) + '.jpg', CROPED_IMAGE)
        # cv2.imshow("Stitched Image", ims_list[0])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Assignment 1!\n")
    print("Affine Puzzle Parts!\n")
    print("Loading...!\n")

    main_()