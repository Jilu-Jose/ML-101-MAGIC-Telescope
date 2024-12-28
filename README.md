#MAGIC Telescope Model
This repository contains a machine learning model trained on the MAGIC Gamma Telescope dataset. This dataset is designed to classify high-energy gamma rays from background noise using information from imaging atmospheric Cherenkov telescopes (IACTs).

Features
Dataset: MAGIC Gamma Telescope dataset, publicly available from the UCI Machine Learning Repository.
Model Type: Random Forest

Input Features:
FLength: Major axis of the ellipse (length)
FWidth: Minor axis of the ellipse (width)
FSize: Size of the ellipse (integrated image intensity)
FConc: Ratio of sum of two highest pixels to the FSize
FConc1: Ratio of the highest pixel to the FSize
FAsym: Distance from the highest pixel to the center
FM3Long: Third root of the third moment along the major axis
FM3Trans: Third root of the third moment along the minor axis
FAlpha: Angle of the major axis with the vector to the origin
FDist: Distance from the origin to the center of the ellipse


Output: Classification of event type:
'1' for gamma rays
'0' for hadrons (background noise)
