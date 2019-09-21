import argparse
import glob
import logging
import multiprocessing as mp
import os
import time
import sys
import cv2

from lib.align_dlib import AlignDlib

logger = logging.getLogger(__name__)

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'lib',  'shape_predictor_68_face_landmarks.dat'))


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    #start_time = time.time()
    #pool = mp.Pool(processes=mp.cpu_count())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_dir in os.listdir(input_dir):
        image_output_dir = os.path.join(output_dir, os.path.basename(os.path.basename(image_dir)))
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)

    image_paths = glob.glob(os.path.join(input_dir, '**/*.jpg'))
    for index, image_path in enumerate(image_paths):
        image_output_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
        output_path = os.path.join(image_output_dir, os.path.basename(image_path))
        preprocess_image(image_path, output_path, args.crop_dim)

    #pool.close()
    #pool.join()
    #logger.info('Completed in {} seconds'.format(time.time() - start_time))


def preprocess_image(input_path, output_path, crop_dim):
    """
    Detect face, align and crop :param input_path. Write output to :param output_path
    :param input_path: Path to input image
    :param output_path: Path to write processed image
    :param crop_dim: dimensions to crop image to
    """
    image = _process_image(input_path, crop_dim)
    if image is not None:
        logger.debug('Writing processed file: {}'.format(output_path))
        cv2.imwrite(output_path, image)
    else:
        logger.warning("Skipping filename: {}".format(input_path))


def _process_image(filename, crop_dim):
    image = None
    aligned_image = None

    image = _buffer_image(filename)

    if image is not None:
        aligned_image = _align_image(image, crop_dim)
    else:
        raise IOError('Error buffering image: {}'.format(filename))

    return aligned_image


def _buffer_image(filename):
    logger.debug('Reading image: {}'.format(filename))
    image = cv2.imread(filename, )
    image = rotate_image(image, 90)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _align_image(image, crop_dim):
    bb = align_dlib.getLargestFaceBoundingBox(image)
    aligned = align_dlib.align(crop_dim, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    if aligned is not None:
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return aligned

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat



def parse_arguments(argv):
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument('--input_dir', type=str, action='store',
                        default="C:\\onesoftdigm\\nodeServer\\node-server\\public\\uploads", dest='input_dir')
    parser.add_argument('--output_dir', type=str, action='store',
                        default="C:\\onesoftdigm\\nodeServer\\node-server\\public\\input", dest='output_dir')
    parser.add_argument('--crop_dim', type=int, action='store', default=180, dest='crop_dim',
                        help='Size to crop images to')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))