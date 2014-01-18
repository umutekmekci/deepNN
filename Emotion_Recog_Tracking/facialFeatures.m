function [bbleye, bbleb, bbreye, bbreb,bbnose, bbaltd,bbustd,bbsold,bbsagd, scaleC] = facialFeatures( I )
%fid3 = figure;

shapeInserter = vision.ShapeInserter('BorderColor','Custom','CustomBorderColor',[255 255 0]);
%face detector
faceDetector = vision.CascadeObjectDetector;
bboxFace = step(faceDetector, I); 
if isempty(bboxFace)
    bbleye = [];  bbleb = 1;  bbreye = 1;  bbreb = 1;  bbnose = 1;  bbaltd = 1;
    bbustd = 1;  bbsold = 1;  bbsagd = 1;  scaleC = 1;
    fprintf('yüz bulunumadý\n');
    return;
end
%bboxFace = bboxFace(1,:); bboxFace(1,2) = bboxFace(1,2)+10;
%faceImage = imcrop(I,bboxFace);
%I_faces = step(shapeInserter, I, int32(bboxFace));    
%set(0,'CurrentFigure',fid3);
%imshow(I_faces);

scaleC = 133/bboxFace(3);
I = imresize(I, scaleC);

faceDetector = vision.CascadeObjectDetector;
bboxFace = step(faceDetector, I); 
bboxFace = bboxFace(1,:); bboxFace(1,2) = bboxFace(1,2)+10;
faceImage = imcrop(I,bboxFace);
I_faces = step(shapeInserter, I, int32(bboxFace));    
%set(0,'CurrentFigure',fid3);
%imshow(I_faces);


mouthDetector = vision.CascadeObjectDetector('Mouth');
bboxMouth = step(mouthDetector, faceImage);
bboxMouth(:,1:2) = bboxMouth(:,1:2) + repmat(bboxFace(1:2),size(bboxMouth,1),1);
I_mouths = step(shapeInserter, I, int32(bboxMouth));    
%set(0,'CurrentFigure',fid3);
%imshow(I_mouths);  
if size(bboxMouth,1) > 1
    [~,ind] = max(bboxMouth(:,2));
    bboxMouth = bboxMouth(ind,:);
end
I_mouths = step(shapeInserter, I, int32(bboxMouth));    
%set(0,'CurrentFigure',fid3);
%imshow(I_mouths);  

mouthImage = imcrop(I, bboxMouth);
[bbustd, bbaltd, bbsold, bbsagd] = findSideLips(mouthImage);
bbustd(1:2) = bbustd(1:2) + bboxMouth(1:2);
bbaltd(1:2) = bbaltd(1:2) + bboxMouth(1:2);
bbsold(1:2) = bbsold(1:2) + bboxMouth(1:2);
bbsagd(1:2) = bbsagd(1:2) + bboxMouth(1:2);
I_mouths = step(shapeInserter, I, int32([bbustd;bbaltd;bbsold;bbsagd]));    
%set(0,'CurrentFigure',fid3);
%imshow(I_mouths);

%hcornerdet = vision.CornerDetector('Method','Local intensity comparison (Rosten & Drummond)');
%pts = step(hcornerdet, rgb2gray(mouthImage));
%hdrawmarkers = vision.MarkerInserter('Shape', 'Circle', 'BorderColor', 'Custom', 'CustomBorderColor', [1,0,0]);
%J = step(hdrawmarkers, mouthImage, pts);
%set(0,'CurrentFigure',fid3);  imshow(J);

%LeftEye detector
leyeDetector = vision.CascadeObjectDetector('LeftEyeCART');
bboxLeye = step(leyeDetector, faceImage);
bboxLeye(:,1:2) = bboxLeye(:,1:2) + repmat(bboxFace(1:2),size(bboxLeye,1),1);
if size(bboxLeye,1) > 1
    [~,ind] = max(bboxLeye(:,1));
    bboxLeye = bboxLeye(ind,:);
end
yuk = bboxLeye(4);
gen = bboxLeye(3);
bboxLeye(1) = bboxLeye(1) + floor((gen/7)*0.5);
bboxLeye(3) = bboxLeye(3) - floor((gen/7)*0.5);
bboxLeye(2) = bboxLeye(2) + floor(yuk/7)*4;
bboxLeye(4) = bboxLeye(4) - floor(yuk/7)*4;
I_leyes = step(shapeInserter, I, int32(bboxLeye));    
%set(0,'CurrentFigure',fid3);
%imshow(I_leyes);  

%LeftEyeBrown
bboxLeb = [bboxLeye(1), bboxLeye(2)-15, bboxLeye(3), floor(bboxLeye(4)/2)];
I_leyebs = step(shapeInserter, I, int32(bboxLeb));    
%set(0,'CurrentFigure',fid3);
%imshow(I_leyebs);  



%RightEye detector
reyeDetector = vision.CascadeObjectDetector('RightEyeCART');
bboxReye = step(reyeDetector, faceImage);
bboxReye(:,1:2) = bboxReye(:,1:2) + repmat(bboxFace(1:2),size(bboxReye,1),1);
if size(bboxReye,1) > 1
    [~,ind] = min(bboxReye(:,1));
    bboxReye = bboxReye(ind,:);
end
yuk = bboxReye(4);
gen = bboxReye(3);
bboxReye(1) = bboxReye(1) + floor((gen/7)*0.5);
bboxReye(3) = bboxReye(3) - floor((gen/7)*0.5);
bboxReye(2) = bboxReye(2) + floor(yuk/7)*4;
bboxReye(4) = bboxReye(4) - floor(yuk/7)*4;
I_reyes = step(shapeInserter, I, int32(bboxReye));    
%set(0,'CurrentFigure',fid3);
%imshow(I_reyes);  

%RightEyeBrown
bboxReb = [bboxReye(1), bboxReye(2)-15, bboxReye(3), floor(bboxReye(4)/2)];
I_reyebs = step(shapeInserter, I, int32(bboxReb));    
%set(0,'CurrentFigure',fid3);
%imshow(I_reyebs);  

%nose detector
noseDetector = vision.CascadeObjectDetector('Nose');
bboxNose = step(noseDetector, faceImage);
bboxNose(:,1:2) = bboxNose(:,1:2) + repmat(bboxFace(1:2),size(bboxNose,1),1);
I_noses = step(shapeInserter, I, int32(bboxNose));    
%set(0,'CurrentFigure',fid3);
%imshow(I_noses);  
len = size(bboxNose,1);
whichnose = false(1,len);
eye_m = bboxReye(2) + floor(bboxReye(4)/2);
mouth_m = bboxMouth(2) + floor(bboxMouth(4)/2);
for i = 1:len
    nose_m = bboxNose(i,:);
    nose_m = nose_m(2) + floor(nose_m(4)/2);
    if nose_m > eye_m && nose_m < mouth_m
        whichnose(i) = true;
    end
end
bboxNose = bboxNose(whichnose,:);
switch sum(whichnose)
    case 0,
        nose_my = eye_m + floor((mouth_m-eye_m)/2);
        nose_mx = bboxMouth(1) + floor(bboxMouth(3)/2);
        gen = bboxLeye(3);  yuk = bboxLeye(4);
        bboxNose = [nose_mx-floor(gen/2), nose_my-floor(yuk/2), gen, yuk];
    otherwise,
        [~,ind] = max(bboxNose(:,2));
        bboxNose = bboxNose(ind,:);
end


yuk = bboxNose(4);
bboxNose(2) = bboxNose(2) + floor(yuk/7)*2;
nose_x2 = bboxNose(2)+bboxNose(4);
if nose_x2 > bboxMouth(2)
    bboxNose(4) = bboxMouth(2) - bboxNose(2) - 10;
end
I_noses = step(shapeInserter, I, int32(bboxNose));    
%set(0,'CurrentFigure',fid3);
%imshow(I_noses);  

bbnose = [bboxNose(2), bboxNose(2)+bboxNose(4); bboxNose(1), bboxNose(1)+bboxNose(3)];
bbnose_m = floor((bbnose(2,1) + bbnose(2,2))/2);
bbaltd = [bbaltd(2), bbaltd(2)+bbaltd(4); bbaltd(1), bbaltd(1)+bbaltd(3)];
bbaltd_m = floor((bbaltd(2,1) + bbaltd(2,2))/2);
bbustd = [bbustd(2), bbustd(2)+bbustd(4); bbustd(1), bbustd(1)+bbustd(3)];
bbustd_m = floor((bbustd(2,1) + bbustd(2,2))/2);
bbsold = [bbsold(2), bbsold(2)+bbsold(4); bbsold(1), bbsold(1)+bbsold(3)];
bbsold_m = floor((bbsold(2,1) + bbsold(2,2))/2);
bbsagd = [bbsagd(2), bbsagd(2)+bbsagd(4); bbsagd(1), bbsagd(1)+bbsagd(3)];
bbsagd_m = floor((bbsagd(2,1) + bbsagd(2,2))/2);

fark = bbnose_m - bbaltd_m;  bbaltd(2,:) = bbaltd(2,:) + [fark, fark];

%bbsold(2,:) = bbsold(2,:) + [fark, fark];
%bbsagd(2,:) = bbsagd(2,:) + [fark, fark];
%bbustd(2,:) = bbustd(2,:) + [fark, fark];

bbleye = [bboxLeye(2), bboxLeye(2)+bboxLeye(4); bboxLeye(1), bboxLeye(1)+bboxLeye(3)];
bbleye_m = floor(mean(bbleye(2,:)));
bbleb = [bboxLeb(2), bboxLeb(2)+bboxLeb(4); bboxLeb(1), bboxLeb(1)+bboxLeb(3)];
bbleb_m = floor(mean(bbleb(2,:)));
bbreye = [bboxReye(2), bboxReye(2)+bboxReye(4); bboxReye(1), bboxReye(1)+bboxReye(3)];
bbreye_m = floor(mean(bbreye(2,:)));
bbreb = [bboxReb(2), bboxReb(2)+bboxReb(4); bboxReb(1), bboxReb(1)+bboxReb(3)];
bbreb_m = floor(mean(bbreb(2,:)));

%fark = bbleye_m - bbsagd_m;  bbsagd(2,:) = bbsagd(2,:) + [fark fark];
%fark = bbreye_m - bbsold_m;  bbsold(2,:) = bbsold(2,:) + [fark fark];

%fark = bbleb_m - bbsagd_m;  bbsagd(2,:) = bbsagd(2,:) + [fark fark];
%fark = bbreb_m - bbsold_m;  bbsold(2,:) = bbsold(2,:) + [fark fark];

bbaltd_t = [bbaltd(2,1),bbaltd(1,1), bbaltd(2,2)-bbaltd(2,1), bbaltd(1,2)-bbaltd(1,1)];
bbsold_t = [bbsold(2,1),bbsold(1,1), bbsold(2,2)-bbsold(2,1), bbsold(1,2)-bbsold(1,1)];
bbsagd_t = [bbsagd(2,1),bbsagd(1,1), bbsagd(2,2)-bbsagd(2,1), bbsagd(1,2)-bbsagd(1,1)];
%bbaltd_t = [bbaltd(2,1),bbaltd(1,1), bbaltd(2,2)-bbaltd(2,1), bbaltd(1,2)-bbaltd(1,1)];
I_mouths = step(shapeInserter, I, int32([bbaltd_t;bbsold_t;bbsagd_t]));    
%set(0,'CurrentFigure',fid3);
%imshow(I_mouths);

end

function [bbustd, bbaltd, bbsold, bbsagd] = findSideLips(mouthImage)
[yuk,gen] = size(mouthImage(:,:,1));
yuko = floor(yuk/2);
geno = floor(gen/2);
if yuk > 10 && gen >10
    stepg = floor(gen/10);
    stepy = floor(yuk/10);
    bbustd = [3*stepg, 0, 5*stepg, 3*stepy];
    bbaltd = [2*stepg, 7*stepy, 6*stepg, 5*stepy];
    bbsold = [-stepg, 2*stepy, 3*stepg, 4*stepy];
    bbsagd = [8*stepg+5, 2*stepy, 3*stepg, 4*stepy];
else
    bbustd = [floor(gen/4) + 2,0, floor(3*(gen/4))-2, yuko-1];
    bbaltd = [floor(gen/4) + 2,yuko+1, floor(3*(gen/4))-2, yuko];
    bbsold = [0,0, floor(gen/4), yuk];
    bbsagd = [floor(3*(gen/4)),0,floor(gen/4), yuk ];
end

end

