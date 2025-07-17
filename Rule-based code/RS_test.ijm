dir_name = "rs"

// Input images path
dir = "./test/";

// Output csv path
csv = "/home/jjyang/jupyter_file/test_speed/predict_RS/"

setBatchMode(true);

//////// The best parameters from grid search: //////////
//default params
aniso = 1;
inRat = 0.1;
maxErr = 1.5000;

// Location of file with all the run times that will be saved:
timeFile = "/home/jjyang/jupyter_file/test_speed/predict_RS/RS_" + dir_name + "_exeTime.txt";

walkFiles(dir);
startTime = getTime();
// Find all files in subdirs:
function walkFiles(dir) {
	list = getFileList(dir);
	for (i=0; i<list.length; i++) {
		if (endsWith(list[i], "/"))
		   walkFiles(""+dir+list[i]);

		// If image file
		else  if (endsWith(list[i], ".tif"))
		   gridProcessImage(dir, csv, list[i]);
	}
}
exeTime = getTime() - startTime; //in miliseconds
// Save exeTime to file:
File.append("Total time:" + exeTime + "\n ", timeFile);

function gridProcessImage(dirPath, csvPath, imName) {

	open("" + dirPath + imName);

    //general params
    var params;
    params = newArray(2, 0.00675, 4, 10);

	//// Just for testing:
	results_csv_path = "" + csvPath + "RS_" + imName  +
	    "_sig" + params[0] +
	    "thr" + params[1] +
	    "suppReg" + params[2] +
	    "intensThr" + params[3] +
	    ".csv";

    RSparams = "image=" + imName +
        " mode=Advanced anisotropy=" + aniso + " use_anisotropy" +
        " robust_fitting=RANSAC" +
        " sigma=" + params[0] +
        " threshold=" + params[1] +
        " support=" + params[2] +
        " min_inlier_ratio=" + inRat +
        " max_error=" + maxErr +
        " spot_intensity_threshold=" + params[3] +
        " background=[No background subtraction]" +
        " results_file=[" + results_csv_path + "]";

	run("RS-FISH", RSparams);
	
	// Close all windows:
	run("Close All");	
	while (nImages>0) { 
		selectImage(nImages); 
		close(); 
    } 
} 
