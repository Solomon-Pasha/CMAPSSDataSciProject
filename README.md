A python data analysis project for the CMAPSS data (https://data.nasa.gov/docs/legacy/CMAPSSData.zip). Aiming to find out what changes in which sensors correlate with engine failure.

To filter out columns which may have no major change over any engines lifespan and thus be effectively irrelevant to finding what foreshadows an engine nearing RUL I used the noise_vs_drift function, which analyses which columns actually change over a engines lifespan.

I used the describe function to generate some basic information on the dataset in the form of the following table
,mean,std,min,max,unique,relStd
operational setting1,-0.0,0.0022,-0.0087,0.0087,158,246.5927
operational setting2,0.0,0.0003,-0.0006,0.0006,13,124.6632
operational setting3,100.0,0.0,100.0,100.0,1,0.0
sensor measurement1,518.67,0.0,518.67,518.67,1,0.0
sensor measurement2,642.6809,0.5001,641.21,644.53,310,0.0008
sensor measurement3,1590.5231,6.1311,1571.04,1616.91,3012,0.0039
sensor measurement4,1408.9338,9.0006,1382.25,1441.49,4051,0.0064
sensor measurement5,14.62,0.0,14.62,14.62,1,0.0
sensor measurement6,21.6098,0.0014,21.6,21.61,2,0.0001
sensor measurement7,553.3677,0.8851,549.85,556.06,513,0.0016
sensor measurement8,2388.0967,0.071,2387.9,2388.56,53,0.0
sensor measurement9,9065.2429,22.0829,9021.73,9244.59,6403,0.0024
sensor measurement10,1.3,0.0,1.3,1.3,1,0.0
sensor measurement11,47.5412,0.2671,46.85,48.53,159,0.0056
sensor measurement12,521.4135,0.7376,518.69,523.38,427,0.0014
sensor measurement13,2388.0962,0.0719,2387.88,2388.56,56,0.0
sensor measurement14,8143.7527,19.0762,8099.94,8293.72,6078,0.0023
sensor measurement15,8.4421,0.0375,8.3249,8.5848,1918,0.0044
sensor measurement16,0.03,0.0,0.03,0.03,1,0.0
sensor measurement17,393.2107,1.5488,388.0,400.0,13,0.0039
sensor measurement18,2388.0,0.0,2388.0,2388.0,1,0.0
sensor measurement19,100.0,0.0,100.0,100.0,1,0.0
sensor measurement20,38.8163,0.1807,38.14,39.43,120,0.0047
sensor measurement21,23.2897,0.1083,22.8942,23.6184,4745,0.0046


Notably the datasets readme.txt file is inconsistent in it's notation, the second number resets between operational setting 3 and sensor measurement 1 but the final sensor measurement is labelled 26 when it should be 21.