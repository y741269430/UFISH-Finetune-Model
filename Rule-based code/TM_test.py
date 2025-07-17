
"""Test script for trackmate's FIJI API."""

import csv
import glob
import os

import time

from ij import IJ
from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.detection import DogDetectorFactory

indir = "./test/"
outdir = "/home/jjyang/jupyter_file/test_speed/predict_TM/"


def run_trackmate(imp, path_out="./", detector="dog", radius=2.5, threshold=0.0, median_filter=True):
    """Log Trackmate detection run with given parameters.
    Saves spots in a csv file in the given path_out with encoded parameters.

    Args:
        imp: ImagePlus to be processed
        path_out: Output directory to save files.
        detector: Type of detection method. Options are 'log', 'dog'.
        radius: Radius of spots in pixels.
        threshold: Threshold value to filter spots.
        median_filter: True if median_filtering should be used.
    """
    if imp.dimensions[2] != 1:
        raise ValueError(
            "Imp's dimensions must be [n, n, 1] but are " + imp.dimensions[2]
        )

    # Create the model object now
    model = Model()
    model.setLogger(Logger.VOID_LOGGER)

    # Prepare settings object
    settings = Settings(imp)

    # Configure detector
    settings.detectorFactory = (
        DogDetectorFactory() if detector == "dog" else LogDetectorFactory()
    )
    settings.detectorSettings = {
        "DO_SUBPIXEL_LOCALIZATION": True,
        "RADIUS": radius,
        "TARGET_CHANNEL": 1,
        "THRESHOLD": threshold,
        "DO_MEDIAN_FILTERING": median_filter,
    }
    trackmate = TrackMate(model, settings)


    # Process
    # output = trackmate.process()
    output = trackmate.execDetection()
    if not output:
        print("error process")
        return None

    # Get output from a single image
    fname = str(imp.title)
    spots = [["axis-0", "axis-1"]]
    for spot in model.spots.iterator(0):
        x = spot.getFeature("POSITION_X")
        y = spot.getFeature("POSITION_Y")
        #q = spot.getFeature("QUALITY")
        spots.append([y, x])

    # Save output
    outname = "TM_" + str(os.path.basename(fname)) + "_r" + str(radius) + "_thr" + str(threshold) + ".csv"
    with open(os.path.join(path_out, outname), "w") as f:
        wr = csv.writer(f)
        for row in spots:
            wr.writerow(row)


## Set parameters
# General params for all test set.
#general_params = {
#    "datasets":{
#        "radius" : 0.1,
#        "threshold" : 0.611590904484 
#    }
#}

### TrackMate
general_params = {
    "datasets":{
        "radius" : 5.0,
        "threshold" : 2.11137767454 
    }
}

files = sorted(glob.glob(os.path.join(indir, "*.tif")))
    
start_time = time.time()

for file in files:
    print("Running: " + file)

    ##general
    params = general_params.get('datasets',general_params)
    
    ##special
    #dataset_name = os.path.basename(file).split("_")[0]
    #params = special_params.get(dataset_name , special_params)

    imp = IJ.openImage(file)
    run_trackmate(imp, path_out=outdir, **params)

end_time = time.time()
total_time = end_time - start_time
print("Total processing time: {}ms ".format(total_time*1000))
