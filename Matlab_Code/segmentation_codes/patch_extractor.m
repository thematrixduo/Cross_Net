%label filename
filename_prefix='P_IM-0005-';
filename_suffix='-0001.txt';
%dicom image filename
img_prefix='IM-0005-';
img_suffix='-0001.dcm';
%patch filename prefix and suffix
patch_img_prefix='A0003-';
patch_label_prefix='AL0003-';
patch_img_suffix='.png';
%labelled patch prefix and suffix
label_img_prefix='L_IM-0005-';
label_img_suffix='-0001.png';
%setting constants
start_index=84;
end_index=164;
patch_size=128;
scaling=0.6;
step_size=4;
steps=2;
shift_vector=zeros((steps*2+1)^2,2,'int16');
counter=1;

%Generate shift vectors
for i=-steps:steps
    for j=-steps:steps
        shift_vector(counter,:)=[i*step_size j*step_size];
        counter=counter+1;
    end
end

%Loops for patch generation
for id=start_index:end_index
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
    marking=table2array(struct2table(tdfread(filename)));
    img_name=strcat(img_prefix,s,img_suffix);
    img_label_name=strcat(label_img_prefix,s,label_img_suffix);
    img=dicomread(img_name);
    img_label=imread(img_label_name);
	%compute center of patch
    min_x=min(marking(:,1))./0.6;
    max_x=max(marking(:,1))./0.6;
    min_y=min(marking(:,2))./0.6;
    max_y=max(marking(:,2))./0.6;
    centre=[int16((min_x+max_x)./2) int16((min_y+max_y)./2)];
	
	%extract patches on shifted centers
    for i=1:size(shift_vector)
        patch_centre=centre+shift_vector(i,:);
        patch=img(patch_centre(2)-patch_size/2+1:patch_centre(2)+patch_size/2,patch_centre(1)-patch_size/2+1:patch_centre(1)+patch_size/2);
        patch(patch(:,:)<0)=0;
        patch=uint16(patch);
        patch_label=img_label(patch_centre(2)-patch_size/2+1:patch_centre(2)+patch_size/2,patch_centre(1)-patch_size/2+1:patch_centre(1)+patch_size/2);
        patch_name=strcat(patch_img_prefix,s,'-',num2str(i),patch_img_suffix);
        patch_label_name=strcat(patch_label_prefix,s,'-',num2str(i),patch_img_suffix);
        imwrite(patch,patch_name);
        imwrite(patch_label,patch_label_name);
    end
        
end 
