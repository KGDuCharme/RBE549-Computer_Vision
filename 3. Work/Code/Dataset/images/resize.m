% What file are we changing? Put name here
name = '61a-NRAX4IL._SL1500_.jpg'

% Read file
image = imread(name);

% Change size
new_image = imresize(image, 0.5);

% Rewrite file with new size
imwrite(new_image,name);

