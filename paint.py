# Written by Manuel Rodriguez Ladron de Guevara, following Aaron Hertzmann's Painterly Rendering with Curved Brush Strokes of Multiple Sizes (SIGGRAPH 98)
import numpy as np
import os
import argparse
import scipy.interpolate as si
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='images/lizard1.jpg')
parser.add_argument('--maxLength', type=int, default=16)
parser.add_argument('--minLength', type=int, default=4)
parser.add_argument('--resize', type=list, default=None)
parser.add_argument('--threshold', type=float, default=0.05)
parser.add_argument('--brush_sizes', type=list, default=[8,4,2])
parser.add_argument('--blur_fac', type=float, default=.5)
parser.add_argument('--grid_fac', type=float, default=1)
parser.add_argument('--length_fac', type=float, default=1)
parser.add_argument('--filter_fac', type=float, default=1)
args = parser.parse_args()

def normalize(img):
    """
    Shifts image from [0-255] to [0-1]
    :param img:
    :return:
    """
    return (img - np.min(img)) / np.ptp(img)

# FRAMEWORK
class Painter():
    """
    """
    def __init__(self, args):
        self.args = args
        self.thres = args.threshold
        self.brush_sizes = args.brush_sizes
        self.maxLength = args.maxLength
        self.minLength = args.minLength
        self.grid_fac = args.grid_fac
        self.len_fac = args.length_fac
        self.filter_fac = args.filter_fac
        self.img, self.canvas = self.get_src_and_canvas(args)
        self.paint(self.img, self.brush_sizes)

    def get_src_and_canvas(self, args):
        """
        Given the args, return numpy image file and canvas (same size as source image)
        :param args:
        :return: source image numpy and canvas (C, H, W)
        """
        #src_img = Image.open(args.image).convert('RGB')
        src_img = cv2.imread(args.image, cv2.IMREAD_COLOR)[:,:,::-1] # from BGR to RGB uint8 [H,W,3]
        imgcv = cv2.imread(args.image)
        self.imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2GRAY) # [H, W]
        if args.resize != None:
            H, W = args.resize[0], args.resize[1]
            self.imgcv = cv2.resize(self.imgcv, (H, W))
            src_img = cv2.resize(src_img, (H,W))
        src_img_np = normalize(src_img).transpose(2,0,1) # [makes ref image in the range [0-1]
        canvas = np.zeros_like(src_img_np) # [3, H, W]
        self.H, self.W = canvas.shape[1], canvas.shape[2]
        return src_img_np, canvas

    def blend(self, canvas, alpha):
        """
        All arrays are in the range [0-1]
        :param canvas: [3, H, W]
        :param alpha: white background alpha of shape [H,W]
        :return:
        """
        color = self.stroke_color[:, None, None] # [3, 1, 1]
        alpha = alpha[None, :, :] # [1, H, W] # white background
        stroke_c = (1 - alpha) * color # black background
        canvas = canvas * alpha + stroke_c
        return canvas

    def bspline(self, cv, n=100, degree=3, periodic=False):
        """ Calculate n samples on a bspline
        https://stackoverflow.com/questions/24612626/b-spline-interpolation-with-python
            cv :      Array ov control vertices
            n  :      Number of samples to return
            degree:   Curve degree
            periodic: True - Curve is closed
                      False - Curve is open
        """
        # If periodic, extend the point array by count+degree+1
        cv = np.asarray(cv)
        count = len(cv)
        if periodic:
            factor, fraction = divmod(count + degree + 1, count)
            cv = np.concatenate((cv,) * factor + (cv[:fraction],))
            count = len(cv)
            degree = np.clip(degree, 1, degree)
        # If opened, prevent degree from exceeding count-1
        else:
            degree = np.clip(degree, 1, count - 1)
        # Calculate knot vector
        kv = None
        if periodic:
            kv = np.arange(0 - degree, count + degree + degree - 1, dtype='int')
        else:
            kv = np.concatenate(([0] * degree, np.arange(count - degree + 1), [count - degree] * degree))
        # Calculate query range
        u = np.linspace(periodic, (count - degree), n)
        # Calculate result
        return np.array(si.splev(u, (kv, cv.T, degree))).T

    def draw_spline(self, K, r):
        alpha = np.zeros([self.canvas.shape[1], self.canvas.shape[2]]).astype('float32') # [H, W]
        pts = self.bspline(K, n=50, degree=3, periodic=False)
        t = 1.
        for p in pts:
            x = int(p[1])
            y = int(p[0])
            cv2.circle(alpha, (x,y), r, t, -1)
        return 1 - alpha

    def makeStroke(self, r, idx_row, idx_col, referenceImage):
        """
        Given starting position [row (idx_x), col (idx_y)] and a stroke max length:
        (1) Iterate over max length pixels
        (2) Calculate gradient (direction, normal, magnitude) at x, y position
        (3) Add dx, dy to list of control pts and update idx_x, idx_y
        :param r: stroke thickness
        :param idx_row: starting position y (row)
        :param idx_col: starting position x (col)
        :param referenceImage:
        :return: list of control points
        """
        ref_color = referenceImage[:, idx_row, idx_col] # R,G,B
        self.stroke_color = ref_color

        canvas_color = self.canvas[:, idx_row, idx_col]
        control_pts = [(idx_row, idx_col)] # Stroke starts here, defined by control points
        last_dx, last_dy = 0,0
        x, y = idx_col, idx_row

        length = int(r * self.len_fac)

        for i in range(self.maxLength):
            # Off boundaries check
            x = max(min(x, self.W-1), 0)
            y = max(min(y, self.H-1), 0)

            # canvas at this point is already properly colored (diff bt canvas and ref is less than ref and stroke)
            if (i > self.minLength) and (np.sqrt(ref_color**2 - canvas_color**2).mean() < np.sqrt(ref_color**2 -
                                                                                           self.stroke_color**2).mean()):
                return control_pts

            # (2) Calculate gradients and its magnitude
            gx, gy = np.sum(self.grad_x[y, x]), np.sum(self.grad_y[y, x])
            g_mag = (gx**2 + gy**2)**0.5

            # Compute normal
            dx, dy = -gy, gx

            # if gradient is small, return control points
            if length*g_mag < 1:
                return control_pts

            # if necessary, reverse direction
            if i > 0 and (last_dx * dx + last_dy * dy) < 0:
                dx, dy = -dx, -dy

            # filter the stroke direction
            dx = (1-self.filter_fac)*last_dx+self.filter_fac*dx
            dy = (1-self.filter_fac)*last_dy+self.filter_fac*dy

            # Compute new magnitude
            g_mag = (dx**2 + dy**2)**0.5

            # New points - "The distance between ctrl points is equal to the brush radius"
            x = int(x + length * dx/g_mag)
            y = int(y + length * dy/g_mag)

            control_pts.append((y, x))  # [H,W]

            # Update last_dx, last_dy
            last_dx = dx
            last_dy = dy

        return control_pts

    def paint_layer(self, canvas, referenceImage, r, threshold):
        """
        Given a stroke thickness r, paint an entire layer using r.
        (1) Divides canvas and refImg into patches of size r
        (2) Iterates over patches
        (3) Calculates initial stroke position
        (4) Makes stroke and appends to stroke layer list
        :param canvas:         [3, H, W] range [0-1]
        :param referenceImage: [3, H, W] range [0-1]
        :param r: constant thickness of stroke
        :param threshold: error map threshold
        :return: list with all strokes in this layer
        """
        strokes = []
        step_size = int(max(r * self.grid_fac, 1))

        # Get intensity of image https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
        # https://www.codementor.io/@innat_2k14/image-data-analysis-using-numpy-opencv-part-1-kfadbafx6
        # https://stackoverflow.com/questions/41971663/use-numpy-to-convert-rgb-pixel-array-into-grayscale

        ref_img = referenceImage.transpose(1,2,0) # [H,W,C]
        rgb_weights = [0.299, 0.587, 0.114]
        img_intensity = np.dot(ref_img[...,:3], rgb_weights) * 1.5 #[H,W]

        self.grad_x = cv2.Sobel(img_intensity,cv2.CV_64F,1,0, ksize=3)    # [H, W]
        self.grad_y = cv2.Sobel(img_intensity, cv2.CV_64F, 0, 1, ksize=3) # [H, W]

        # Iterate over each patch, calculate error and find largest error in x,y coord
        xx = 0
        for x in range(0, self.W, step_size): # iterating over x axis (columns)
            yy = 0
            for y in range(0, self.H, step_size): # iterating over y axis (rows)
                # Get error in the patch (cell)
                patch_ref = referenceImage[:, y:min(y+step_size, self.H-1), x:min(x+step_size, self.W-1)]
                patch_canvas = canvas[:, y:min(y + step_size, self.H-1), x:min(x + step_size, self.W-1)]

                patch_loss = np.abs(patch_canvas - patch_ref) # L1 Loss
                patch_loss_ = patch_loss.mean(0)  # Mean across channels -> [H, W] patch
                area_error = patch_loss.mean() # Scalar
                #print(f'Error patch: {area_error}. Threshold: {threshold}')

                if area_error > threshold:
                    # Find x,y position for stroke based on max error
                    fidx_row, fidx_col = np.unravel_index(np.argmax(patch_loss_), patch_loss_.shape)
                    row_abs = fidx_row + (yy * patch_loss_.shape[0])
                    col_abs = fidx_col + (xx * patch_loss_.shape[1])

                    # Make stroke and blend
                    stroke = self.makeStroke(r, row_abs, col_abs, referenceImage)  # List of control points
                    alpha = self.draw_spline(stroke, r)  # white background
                    self.canvas = self.blend(self.canvas, alpha)  # [3, H, W]

                    strokes.append(stroke)
                yy += 1
            xx += 1
        return strokes

    def paint(self, source_img, brush_sizes):
        """
        :param source_img: numpy as array as shape [C, H, W]
        :param brush_sizes: python list with number of different brush radii
        :return:
        """
        for brush_size in brush_sizes:

            blur_fac = int(self.args.blur_fac * brush_size)
            if blur_fac % 2 == 0: # Handle even factors
                blur_fac += 1

            print(f'\nLayer corresponding to brush size: {brush_size}')
            print(f'Blurring reference image with a blur factor of {blur_fac}')

            if blur_fac > 0:
                ref_img = cv2.GaussianBlur(source_img.transpose(1, 2, 0), (blur_fac, blur_fac), 0)
                ref_img = ref_img.transpose(2,0,1) # [C, H, W]
            else:
                ref_img = source_img

            # Paint layer
            strokes_layer = self.paint_layer(self.canvas, ref_img, brush_size, threshold=self.thres)
            print(f'\n At layer bsize {brush_size}, there are {len(strokes_layer)} strokes')
            canvas_save = self.canvas.transpose(1, 2, 0)[:, :, ::-1] * 255.
            namefile = os.path.basename(self.args.image)
            cv2.imwrite(f'{namefile}_level_{brush_size}.jpeg', canvas_save)  # [H,W,C]

Painter(args)