#!/usr/bin/env python2

import argparse

import facetracker
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PIL import Image


def rm_dir_and_ext(filepath):
    return filepath.split('/')[-1].split('.')[-2]


def save_plot(portrait, points, name, out_dir="./"):
    fig = Figure()
    fig.set_dpi(50)
    fig.set_size_inches(12, 16)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.imshow(portrait)
    ax.scatter(points[:, 0], points[:, 1], s=50)
    out_path = out_dir + name + "_points.png"
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)


def get_tracker_points(args):
    """
    FaceTracker settings:
    - clamp      : [0-4] 1 gives a very loose fit, 4 gives a very tight
                   fit
    - iterations : Number of iterations. Should be a number in the
                   range: 1-25, 1 is fast and inaccurate, 25 is slow and
                   accurate
    - tolerance  : Matching tolerance. Should be a double in the range:
                   .01-1.

    pyFaceTracker:
    http://pythonhosted.org/pyFaceTracker/generated/facetracker.FaceTracker.html

    FaceTracker:
    https://github.com/kylemcdonald/FaceTracker
    """

    portrait_path = args.portrait_path
    tracker_path = args.tracker_path
    clamp = args.clamp
    iterations = args.iterations
    tolerance = args.tolerance
    out_dir = args.output_dir
    plot = args.plot

    # Load portrait
    portrait = Image.open(portrait_path)
    portrait_gray = portrait.convert("L")
    portrait_gray_np = np.array(portrait_gray)

    # Load tracker
    tracker = facetracker.FaceTracker(tracker_path)

    # Get tracker points
    tracker.resetFrame()
    if clamp:
        tracker.clamp = clamp
    tracker.iterations = iterations
    if tolerance:
        tracker.tolerance = tolerance
    tracker.update(portrait_gray_np)

    # Format tracker points
    malformed_points = tracker.get2DShape()[0]
    nb_coords = len(malformed_points)
    nb_points = nb_coords / 2
    xs = malformed_points[:nb_points]
    ys = malformed_points[nb_points:]
    points = zip(xs, ys)
    points_np = np.squeeze(np.array(points))

    # Save tracker points
    name = rm_dir_and_ext(portrait_path)
    np.save(out_dir + name + ".npy", points_np)

    # Output plot
    if plot:
        save_plot(portrait, points_np, name, out_dir=out_dir)


if __name__ == '__main__':
    tracker_path = "/nbs/celeba/FaceTracker/model/face.tracker"

    tracker_help = "e.g. " + tracker_path
    clamp_help = "[0-4] 1 gives a very loose fit, 4 gives a very " + \
                 "tight fit"
    iterations_help = "Number of iterations. Should be a number " + \
                      "in the range: 1-25, 1 is fast and " + \
                      "inaccurate, 25 is slow and accurate"
    tolerance_help = "Matching tolerance. Should be a double in " + \
                     "the range: .01-1."

    parser = argparse.ArgumentParser()
    parser.add_argument("portrait_path")
    parser.add_argument("-tr", "--tracker_path", help=tracker_help,
                        default=tracker_path)
    parser.add_argument("-c", "--clamp", help=iterations_help,
                        default=None)
    parser.add_argument("-i", "--iterations", help=iterations_help,
                        default=25)
    parser.add_argument("-to", "--tolerance", help=tolerance_help,
                        default=None)
    parser.add_argument("-o", "--output_dir",
                        default="./portraitseg/outputs/")
    parser.add_argument("-p", "--plot", action='store_true')
    args = parser.parse_args()

    portrait_path = args.portrait_path
    tracker_path = args.tracker_path
    clamp = args.clamp
    tolerance = args.tolerance
    iterations = args.iterations
    out_dir = args.output_dir

    get_tracker_points(args)
