%Patch extractor for image classification
%right click=negative left click=positive
%middle button=terminate


%reading saved count file if exist
if exist('count.mat','file')>0
    load('count.mat');
else
    start_from=1;
    patch_count_pos=0;
    patch_count_neg=0;
end
%number of images to be read
num_image=140;
terminate=0;

for id=start_from:num_image
    s=num2str(id)
    if (id<100)
        s=strcat('0',s)
        if (id<10)
            s=strcat('0',s)
        end
    end
	%read image
    filename=strcat('a',s,'.bmp');
    img=rgb2gray(imread(filename));
    patch_half_width=64;
    figure;
    imshow(img);
	%get mouse clicks on image
    [x,y,button]=ginput;
    for i=1:length(x)
        if button(i)==2 
            terminate=1;
            close all;
            break;
        end
        x_coor=round(x(i));
        y_coor=round(y(i));
        patch=imcrop(img,[x_coor-patch_half_width,y_coor-patch_half_width,patch_half_width*2-1,patch_half_width*2-1]);
        figure;
        new_patch=imresize(patch,0.25,'Method','bilinear');
        imshow(patch);
        figure;
        imshow(new_patch);
        
        if button(i)==1 
            patch_count_neg=patch_count_neg+1;
            patch_name=strcat('n-',num2str(patch_count_neg),'.png');
        elseif button(i)==3
            patch_count_pos=patch_count_pos+1;
            patch_name=strcat('p-',num2str(patch_count_pos),'.png');
        end
        imwrite(new_patch,patch_name);
        close all;
    end    
    if terminate==1
        disp(sprintf('Patch extraction process terminated at image with name: %s',filename));
        break;
    end
end
start_from=id+1;
save('count.mat','patch_count_pos','patch_count_neg','start_from');