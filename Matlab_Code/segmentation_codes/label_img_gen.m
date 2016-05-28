%set filenames, indexes and scaling
filename_prefix='P_IM-0005-';
filename_suffix='-0001.txt';
cmp_filename_prefix='IM-0005-';
cmp_filename_suffix='-0001.cmp';
label_img_prefix=cmp_filename_prefix;
label_img_suffix='-0001.png';
start_index=84;
end_index=164;
scaling=0.6;

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
    cmp_filename=strcat(cmp_filename_prefix,s,cmp_filename_suffix);
    cmp_notation=fileread(cmp_filename);
    %check if there is thrombus
    thrombus_found=strfind(cmp_notation,'Thrombus');
    %convert label coordinates to pixel maps
    marking=table2array(struct2table(tdfread(filename)));
    lumen=marking(marking(:,4)==1,1:2)./0.6;
    lumen=[lumen;lumen(1,:)];
    wall=marking(marking(:,4)==2,1:2)./0.6;
    wall=[wall;wall(1,:)];
    other=[marking(marking(:,4)>2,1:2)./0.6 marking(marking(:,4)>2,4)];
    wall_fill=2*uint16(poly2mask(wall(:,1),wall(:,2),512,512));
    lumen_fill=uint16(poly2mask(lumen(:,1),lumen(:,2),512,512));
    
    stop=0;
    counter=3;
    %convert to pixel maps for thrombus class
    if ~isempty(thrombus_found)
        for i=1:size(thrombus_found,2)
            thrombus=other(other(:,3)==counter,1:2);
            thrombus=[thrombus;thrombus(1,:)];
            thrombus_fill=uint16(poly2mask(thrombus(:,1),thrombus(:,2),512,512));
            wall_fill(thrombus_fill(:,:)==1)=4;
            counter=counter+1;
        end
    end
    wall_fill(lumen_fill(:,:)==1)=1;
    %convert for calcium class
    while stop==0
        calcium_single=other(other(:,3)==counter,1:2);
        if isempty(calcium_single)
            stop=1;
        else
            calcium_single=[calcium_single;calcium_single(1,:)];
            calcium_fill=uint16(poly2mask(calcium_single(:,1),calcium_single(:,2),512,512));
            wall_fill(calcium_fill(:,:)==1)=3;
        end
        counter=counter+1;
    end
    %write label pixel map
    label_img_filename=strcat('L_',label_img_prefix,s,label_img_suffix);
    imwrite(wall_fill,label_img_filename);
end