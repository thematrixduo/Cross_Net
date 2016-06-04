%set filenames, indexes and scaling
filename_prefix='IM-0005-';
filename_suffix='-0001.dcm';
cmp_filename_prefix='IM-0005-';
cmp_filename_suffix='-0001.cmp';
patch_prefix='patch-0005-';
patch_suffix='-0001.png';
patch_size_half=64;
start_index=84;
end_index=164;
filter_width=25;
filter=fspecial('disk',filter_width);
threshold_fac=0.55;
%loop through images
for id=start_index:end_index
    %convert iterator to string
    s=num2str(id);
    if (id<1000)
        s=strcat('0',s);
        if (id<100)
            s=strcat('0',s);
            if (id<10)
                s=strcat('0',s);
            end
        end
    end
    %read files
    filename=strcat(filename_prefix,s,filename_suffix);
    img=dicomread(filename);
    %thresholding
    max_val=max(max(img));
    img_salient=img;
    img_salient(img_salient<max_val*threshold_fac)=0;
    %convolve with disk filter
    img_filtered=imfilter(img_salient,filter,'replicate');
    %figure;
    %imshow(img,[0,2000]);
    %figure;
    %imshow(img_salient,[0,2000]);
    %figure;
    %imshow(img_filtered,[0,2000]);
    %search for maximum value index
    [maxA,ind] = max(img_filtered(:));
    [m,n] = ind2sub(size(img_filtered),ind);
    patch=img(m-patch_size_half:m+patch_size_half-1,n-patch_size_half:n+patch_size_half-1);
    patch=uint16(patch);
    %figure;
    %imshow(patch,[0,2000]);
    patch_filename=strcat(patch_prefix,s,patch_suffix);
    imwrite(patch,patch_filename);
end