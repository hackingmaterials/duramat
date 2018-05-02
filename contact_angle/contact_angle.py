

import os
import operator
import collections
import itertools

import numpy as np
from scipy.ndimage import gaussian_filter

from skimage import io, color, feature, morphology, filters, exposure, restoration, segmentation

import matplotlib.pyplot as plt


def main():
    pass


class ContactAngle(object):

    def __init__(self, img):
        """Initialize object with image.

        Args:
            img (np.ndarray, str): Image as array or /path/to/file
        """
        # if os.path.isfile(img) and isinstance(str, img):
        #     self.img = io.imread(img, as_grey=True)
        if False:
            pass
        elif isinstance(img, np.ndarray):
            if len(img.shape) > 2:
                self.img = img
                # self.img = color.rgb2gray(img)
            else:
                self.img = img.copy()
        else:
            raise ValueError("Must pass image (as numpy array) or /path/to/file.")

        self.img_processed = None
        self.labels = None
        self.measured_parameters = None

    def crop(self):
        """Crop image.

        Returns:
            None
        """
        shape_x, shape_y = self.img.shape[0], self.img.shape[1]
        x_lo, x_hi = int(shape_x * .2), int(shape_x * .8)
        y_lo, y_hi = int(shape_y * .2), int(shape_y * .8)
        img2 = self.img[x_lo: x_hi, y_lo: y_hi]
        self.img = img2.copy()

    def show(self, title='Processed'):
        """Plot image (matplotlib).

        args:
            title (str): title for image

        Returns:
            None
        """
        if self.img_processed is None:
            f, a = plt.subplots(figsize=(18, 6))
            a.imshow(self.img, cmap="Greys_r")
        else:
            f, a = plt.subplots(ncols=2, figsize=(18, 6))
            a[0].imshow(self.img, cmap="gray")
            a[0].set_title('Original')

            a[1].imshow(self.img, cmap="gray")
            # a[1].imshow(self.img_processed, cmap="gray")
            if self.labels is not None:
                indices = np.argwhere(self.labels > 0)
                a[1].scatter(indices[:, 1], indices[:, 0], c='r', s=1)
            if self.measured_parameters is not None:
                d = self.measured_parameters
                a[1].scatter(*d['p1'], c='b', s=50, edgecolor='w')
                a[1].scatter(*d['p2'], c='g', s=50, edgecolor='w')
                a[1].scatter(*d['p1_b'], c='b', s=50, edgecolor='w')
                a[1].scatter(*d['p2_b'], c='g', s=50, edgecolor='w')
                a[1].scatter(*d['midpt'], c='y', s=50, edgecolor='w')
                a[1].scatter(*d['top_pt'], c='y', s=50, edgecolor='w')
                a[1].plot([d['p1'][0], d['p2'][0]], [d['p1'][1], d['p2'][1]], c='w')
                a[1].plot([d['midpt'][0], d['top_pt'][0]], [d['midpt'][1], d['top_pt'][1]], c='w')
                text = 'Left angle: {:.3f}$^o$\nRight angle: {:.3f}$^o$\nBase: {:.3f}\nHeight: {:.3f}'\
                    .format(d['Left contact angle'], d['Right contact angle'], d['Width'], d['Height'])
                a[1].text(50, 200, text, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5}, fontdict={'size': 12})
            a[1].set_title(title)
            f.tight_layout()

    def get_input_image(self):
        """Decide which image should be processed (default or a 'workspace' image).

        Returns:
            np.ndarray: image to process.
        """
        input_img = self.img if self.img_processed is None else self.img_processed
        return input_img

    def canny_edge(self, sigma=1):
        """Canny edge detection on image.

        Args:
            sigma (float):  parameter for canny edge (from skimage)

        Returns:
            None
        """
        input_img = self.get_input_image()

        self.img_processed = feature.canny(input_img, sigma=sigma)

    def roberts_edge(self):
        """Perform roberts edge detection.

        Returns:
            None
        """
        input_img = self.get_input_image()

        self.img_processed = filters.roberts(input_img)

    def dilation(self, structure=np.ones((3, 3))):
        """Perform dilation on image.

        Args:
            structure (np.ndarray): structure for dilation.

        Returns:
            None
        """
        input_img = self.get_input_image()

        self.img_processed = morphology.dilation(input_img, structure)

    def closing(self, structure=np.ones((3, 3))):
        """Perform closing.

        Args:
            structure (np.ndarray):  structure for closing

        Returns:
            None
        """
        input_img = self.get_input_image()

        self.img_processed = morphology.closing(input_img, structure)

    def label_regions(self):
        """Find continously connected regions.

        Returns:
            None
        """
        if self.img_processed is None:
            raise RuntimeError("Labelling raw image isn't allowed.")
        input_img = self.img_processed
        labels = morphology.label(input_img)
        region_sizes = collections.Counter(labels.flatten())
        region_sizes = {key: value for key, value in region_sizes.items() if key > 0}
        largest_region = max(region_sizes.items(), key=operator.itemgetter(1))[0]
        labels[labels != largest_region] = 0
        labels[labels == largest_region] = 1
        self.labels = labels

    def furthest_two_points(self):
        """Find furthest two points in an array.  Points to consider must have values = 1.  All other elements
        should be 0.

        Returns:
            tuple: coordinates of two points, ordered by x-value
        """
        furthest = -1
        furthest_i, furthest_j = None, None
        contour = np.argwhere(self.labels >= 1)
        low_pct, high_pct = np.percentile(contour[:, 1], 5), np.percentile(contour[:, 1], 95)
        contour_low = contour[contour[:, 1] < low_pct]
        contour_high = contour[contour[:, 1] > high_pct]
        for x in itertools.product(contour_low, contour_high):
            i, j = x
            dist = np.linalg.norm(i - j)
            if dist > furthest:
                furthest = dist
                furthest_i = i
                furthest_j = j

        p1 = (furthest_i[1], furthest_i[0])
        p2 = (furthest_j[1], furthest_j[0])

        if p1[0] < p2[0]:
            return p1, p2
        else:
            return p2, p1

    def calc_midpt(self, p1, p2):
        """Calculate midpoint between two points.

        Args:
            p1 (tuple): (x, y) of first point
            p2 (tuple): (x, y) of second point

        Returns:
            tuple: (x, y) of midpoint
        """
        midpt = [(p1[0] + p2[0]) // 2, (p1[1] + p2[1]) //2]
        return midpt

    def equalize_hist(self):
        input_img = self.get_input_image()

        self.img_processed = exposure.equalize_adapthist(input_img)

    def erosion(self, structure=np.ones((3, 3))):
        input_img = self.get_input_image()

        self.img_processed = morphology.erosion(input_img, selem=structure)

    def adjust_gamma(self, gamma=1):
        input_img = self.get_input_image()

        self.img_processed = exposure.adjust_gamma(input_img, gamma=gamma)

    def adjust_histogram(self, gamma=1):
        input_img = self.get_input_image()

        self.img_processed = exposure.equalize_hist(input_img, nbins=100)

    def extract(self):
        """Extract contact angles, height, and width of droplet in image

        Returns:
            dict: measured parameters

            keys of most interest are:
                'Left contact angle', 'Right contact angle', 'Width', 'Height'

            other keys used to derive these parameters are:
                'p1', 'p1_b' -- used to calculate left contact angle
                'p2', 'p2_b' -- used to calculate right contact angle
                'midpt', 'top_pt' -- used to calculate contact angles, width, and height
        """
        p1, p2 = self.furthest_two_points()
        midpt = self.calc_midpt(p1, p2)

        p1_by = p1[1] - 20
        p1_bx = np.argmax(self.labels[p1_by, :])
        p1_b = (p1_bx, p1_by)

        p2_by = p2[1] - 20
        p2_bx = self.labels.shape[1] - np.argmax(self.labels[p2_by, :][::-1])
        p2_b = (p2_bx, p2_by)

        p1_angle = self.angle_between_lines(p1, midpt, p1_b)
        p2_angle = self.angle_between_lines(p2, midpt, p2_b)

        top_pt = self.find_90_deg_point(midpt, p2)
        height = self.calc_dist(top_pt, midpt)
        length = self.calc_dist(p1, p2)

        self.measured_parameters = {'p1': p1, 'p1_b': p1_b, 'Left contact angle': p1_angle,
                                    'p2': p2, 'p2_b': p2_b, 'Right contact angle': p2_angle,
                                    'midpt': midpt, 'top_pt': top_pt, 'Height': height, 'Width': length}

        return self.measured_parameters

    def angle_between_lines(self, p0, p1, p2):
        """Calculate angle between three points.

        Args:
            p0 (tuple): (x, y) of vertex
            p1 (tuple): (x, y) of non-vertex
            p2 (tuple): (x, y) of non-vertex

        Returns:
            float: angle (in degrees)
        """
        p0 = np.asarray(p0)
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        p1_p0 = p1 - p0
        p2_p0 = p2 - p0

        cos_theta = np.dot(p1_p0, p2_p0) / (np.linalg.norm(p1_p0) * np.linalg.norm(p2_p0))
        return np.degrees(np.arccos(cos_theta))

    def find_90_deg_point(self, p0, p1):
        """Find point closest to 90 degrees from two other points.

        Args:
            p0 (tuple): (x, y) of vertex
            p1 (tuple): (x, y) of non-vertex

        Returns:
            tuple: (x, y) of point
        """
        best_angle = 1e6
        best_p = None
        values = np.argwhere(self.labels >= 1)
        for p2 in values:
            p2 = p2[::-1]
            angle = self.angle_between_lines(p0, p1, p2)
            if np.abs(angle - 90) < best_angle and p2[1] < p1[1]:
                best_angle = np.abs(angle - 90)
                best_p = p2
        return best_p

    def calc_dist(self, p1, p2):
        """Calculate distance between two points.

        Args:
            p1 (tuple): (x, y) of point 1
            p2 (tuple): (x, y) of point 2

        Returns:
            float: distance
        """
        return np.linalg.norm(np.asarray(p1) - np.asarray(p2), 2)

    def run_pipeline(self, sigma=1):
        """Run 'current' contact angle extraction pipeline.

        Args:
            sigma (float):  sigma value for canny edge detection

        Returns:
            dict: parameters of contact angle -- see extract() docs
        """
        self.crop()
        self.canny_edge(sigma=sigma)
        self.dilation(morphology.disk(3))
        self.label_regions()
        params = self.extract()
        self.show(title='Final')
        return params

    def _explore_func(self):
        """Function used for exploratory analysis.  There should be *NO* expectation that this function provides
        consistent functionality.

        Returns:
            None
        """
        # self.crop()
        # self.img_processed = self.img[:, :, 0]
        # self.img_processed = filters.rank.minimum(
        #     filters.rank.maximum(self.img, morphology.disk(5)), morphology.disk(5))
        # self.img_processed = filters.rank.enhance_contrast_percentile(self.img, morphology.disk(5), p0=.2, p1=.5)
        # self.img_processed = filters.rank.enhance_contrast(self.img, morphology.disk(3))
        # self.img_processed = filters.rank.mean_bilateral(self.img, np.ones((3, 3)))
        # self.img_processed = filters.rank.mean_bilateral(self.img, morphology.disk(5))
        # self.img_processed = filters.rank.mean_bilateral(self.img, morphology.disk(10))
        # self.img_processed = filters.rank.(self.img, morphology.disk(10))
        # self.img_processed = exposure.equalize_hist(self.img, nbins=128)
        # self.img_processed = filters.threshold_otsu(self.img)
        # self.img_processed = filters.rank.equalize(self.img, morphology.disk(50))

        # image = gaussian_filter(self.img, .1)
        # image = self.img.copy()

        # seed = np.copy(image)
        # seed[1:-1, 1:-1] = image.min()
        # mask = image

        # self.img_processed = morphology.reconstruction(seed, mask, method='dilation', selem=morphology.disk(3))


        self.canny_edge(sigma=5)
        self.dilation(np.ones((10, 10)))
        # self.dilation(np.ones((3, 3)))
        # self.show()
        # self.dilation(morphology.disk(140))
        # self.show()
        # self.dilation()
        f, a = plt.subplots(ncols=2, figsize=(18, 6))
        a[0].imshow(self.img, cmap='gray')
        a[1].imshow(self.img_processed, cmap='gray')
        # self.label_regions()
        # self.extract()
        # self.show()

if __name__ == '__main__':
    main()
