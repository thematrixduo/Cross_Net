%transform label maps to coloured image
filename='AL0003-0163-13.png';
label_img=imread(filename);
figure;
colour_map=[0 0 0;1.0 0 0;0 1.0 0;0 0 1.0;0.5 0 0.5];
imshow(label_img,colour_map);