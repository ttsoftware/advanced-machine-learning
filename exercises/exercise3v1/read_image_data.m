I = imresize(im2double(imread('flower.png')), 0.1);
data = reshape(I, [size(I, 1)*size(I, 2), 3]);
