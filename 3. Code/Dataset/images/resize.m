name = '61a-NRAX4IL._SL1500_.jpg'

image = imread(name);

new_image = imresize(image, 0.5);
% imshow(new_image)

imwrite(new_image,name);

