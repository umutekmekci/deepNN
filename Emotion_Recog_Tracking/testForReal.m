function [ output ] = testForReal

output = 1;
test = false;

hgauss = fspecial('gaussian',5,0.7);
m = [0.2170, 1.9337, -0.9441; -3.2572, 3.2572, 0.5428; 3.2872, -3.3572, 0.5318];

vidobj = videoinput('winvideo',1,'RGB24_320x240');
triggerconfig(vidobj, 'manual');
start(vidobj);
pause(3);
if test
    image = getsnapshot(vidobj);  % imshow(image);
    image = imfilter(image, hgauss);
    [bbleye, bbleb, bbreye, bbreb, bbnose, bbaltd,bbustd,bbsold,bbsagd] = facialFeatures(image);
    image = double(image);
    imor = image;  
    iR = image(:,:,1);  iG = image(:,:,2);  iB = image(:,:,3);
    iMat = [iR(:)';iG(:)';iB(:)'];
    iMat = m*iMat;
    image(:,:,1) = 1.5*reshape(iMat(1,:),size(iR));
    image(:,:,2) = reshape(iMat(2,:),size(iR));
    image(:,:,3) = 1.6*reshape(iMat(3,:),size(iR));
    image(image>255) = 255;  image(image<0) = 0 ;
    [bbreb,bbleb,bbreye,bbleye] = imageAnalyzeGenel( image,bbreb,bbleb,bbreye,bbleye,3,2 );
    [bbaltd, bbsagd, bbsold, bbnose] = imageAnalyzeMouth(imor,bbaltd,bbsagd, bbsold, bbnose, bbleye, bbreye, 3,image);
    imshow(uint8(imor));  hold on;
    bb=boundingBox(bbnose);
    plot(bb(2,:),bb(1,:),'b.');
    bb=boundingBox(bbleye);
    plot(bb(2,:),bb(1,:),'b.');
    bb=boundingBox(bbleb);
    plot(bb(2,:),bb(1,:),'b.');
    bb=boundingBox(bbreye);
    plot(bb(2,:),bb(1,:),'b.');
    bb=boundingBox(bbreb);
    plot(bb(2,:),bb(1,:),'b.');
    bb=boundingBox(bbaltd);
    plot(bb(2,:),bb(1,:),'b.');
    bb=boundingBox(bbsagd);
    plot(bb(2,:),bb(1,:),'b.');
    bb=boundingBox(bbsold);
    plot(bb(2,:),bb(1,:),'b.');
    stop(vidobj);
    delete(vidobj);
end



t = 3;
kuyruk = zeros(4,t);

matobj = matfile('C:\Users\daredavil\Documents\MATLAB\Tracking\SVMSML\ldas');
WLDA =  matobj.W;
clear matobj;

matobj = matfile('C:\Users\daredavil\Documents\MATLAB\Tracking\SVMSML\happy');
SVMStructHappy = matobj.svm;
clear matobj;

matobj = matfile('C:\Users\daredavil\Documents\MATLAB\Tracking\SVMSML\thinking');
SVMStructThinking = matobj.svm;
clear matobj;

matobj = matfile('C:\Users\daredavil\Documents\MATLAB\Tracking\SVMSML\agree');
SVMStructAgree = matobj.svm;
clear matobj;

matobj = matfile('C:\Users\daredavil\Documents\MATLAB\Tracking\SVMSML\suprise');
SVMStructSuprise = matobj.svm;
clear matobj;

matobj = matfile('C:\Users\daredavil\Documents\MATLAB\Tracking\SVMSML\disagree');
SVMStructDisagree = matobj.svm;
clear matobj;

matobj = matfile('C:\Users\daredavil\Documents\MATLAB\Tracking\SVMSML\anger');
SVMStructAnger = matobj.svm;
clear matobj;

matobj = matfile('C:\Users\daredavil\Documents\MATLAB\Tracking\SVMSML\normal');
SVMStructNormal = matobj.svm;
clear matobj;

matobj = matfile('C:\Users\daredavil\Documents\MATLAB\Tracking\featureVecs\peakSML');
peakV = matobj.P;
clear matobj;

matobj = matfile('C:\Users\daredavil\Documents\MATLAB\Tracking\SVMML2\nnet3');
nnet = matobj.nnet;
clear matobj;

matobj = matfile('C:\Users\daredavil\Documents\MATLAB\Tracking\SVMML2\tree3');
tree = matobj.tree;
clear matobj;

image = getsnapshot(vidobj);
image = imfilter(image, hgauss);
[bbleye, bbleb, bbreye, bbreb, bbnose, bbaltd,bbustd,bbsold,bbsagd,scaleC] = facialFeatures(image);
if isempty(bbleye)
    stop(vidobj);
    delete(vidobj);
end

fark = floor(mean(bbleye(1,:))) - floor(mean(bbleb(1,:)));
image = imresize(image, scaleC);
F = zeros(4,1);
nosebefore = 0;
xmb = 0;
ymb = 0;
leftC = 0;
rightC = 0;
upC = 0;
downC = 0;
solCC = 0;
sagCC = 0;
farklips2 = 0;
farklips2son = 0;
farklips = 0;
farklipsson = 0;
SSB = 0;
tekrar = 3;
sayac = 1;
iter = 1;
kararV = zeros(1,7);
S{1} = 'happy';  S{2} = 'thinking';  S{3} = 'agree'; S{4} = 'disagree';
S{5} = 'anger';  S{6} = 'suprise';  S{7} = 'normal';
maxiter = 1000;

while iter < maxiter
    image = double(image);   imshow(uint8(image));  drawnow;
    imor = image;  
    iR = image(:,:,1);  iG = image(:,:,2);  iB = image(:,:,3);
    iMat = [iR(:)';iG(:)';iB(:)'];
    iMat = m*iMat;
    image(:,:,1) = 1.5*reshape(iMat(1,:),size(iR));
    image(:,:,2) = reshape(iMat(2,:),size(iR));
    image(:,:,3) = 1.6*reshape(iMat(3,:),size(iR));
    image(image>255) = 255;  image(image<0) = 0 ;
    [bbreb,bbleb,bbreye,bbleye] = imageAnalyzeGenel( image,bbreb,bbleb,bbreye,bbleye,3,2 );
    xmn = floor(mean(bbleye(1,:)));  ymn = floor(mean(bbleye(2,:)));
    farkt = ( ( floor(mean(bbleye(1,:))) - floor(mean(bbleb(1,:))) ) + ( floor(mean(bbreye(1,:))) - floor(mean(bbreb(1,:))) ) )/2;
    [up,down,right,left] = findYon(xmb,ymb, xmn, ymn);
    if up && right
        sagCC = sagCC + 1;
    else
        sagCC = max(0,sagCC-1);
    end
    if up && left
        solCC = solCC + 1;
    else
        solCC = max(0,solCC-1);
    end
    if ~up && ~down && left
        leftC = leftC + 1;
    else
        leftC = max(0,leftC-1);
    end
    if ~up && ~down && right
        rightC = rightC + 1;
    else
        rightC = max(0,rightC-1);
    end
    if ~left && ~right && up
        upC = upC + 1;
    else
        upC = max(0,upC-1);
    end
    if ~left && ~right && down
        downC = downC + 1;
    else
        downC = max(0,downC-1);
    end
    xmb = xmn;  ymb = ymn;
    
    
    
    [bbaltd, bbsagd, bbsold, bbnose] = imageAnalyzeMouth(imor,bbaltd,bbsagd, bbsold, bbnose, bbleye, bbreye, 3,image);
    farklipst = floor(mean(bbsagd(2,:))) - floor(mean(bbsold(2,:)));
    farklips2t = floor(mean(bbaltd(1,:))) - floor(mean(bbnose(2,:)));
    
    
    
    rebm = floor(mean(bbreb,2));  lebm = floor(mean(bbleb,2)); reym = floor(mean(bbreye,2));  leym = floor(mean(bbleye,2));
    altdm = floor(mean(bbaltd,2));  sagdm = floor(mean(bbsagd,2));  soldm = floor(mean(bbsold,2));  nosem = floor(mean(bbnose,2));
    
    
    
    
    F(1,:) = reym(1) - rebm(1);
    F(2,:) = leym(1) - lebm(1);
    F(3,:) = soldm(2) - sagdm(2);
    F(4,:) = altdm(1) - floor((soldm(1)+sagdm(1))/2);
    SS = [altdm(2);reym(2);altdm(1);reym(1)];
    
    %F(1:2) = nosem - rebm;
    %F(3:4) = nosem - reym;
    %F(5:6) = nosem - lebm;
    %F(7:8) = nosem - leym;
    %F(9:10) = nosem - altdm;
    %F(11:12) = nosem - sagdm;
    %F(13:14) = nosem - soldm;
    %F = F./(leym(2)-reym(2));
    %F(15:16) = nosem - nosebefore;   nosebefore = nosem;
    
    kuyruk(:,2:end) = kuyruk(:,1:end-1);  kuyruk(:,1) = F;
    FV = [kuyruk(:); SS - SSB];  SSB = SS;  FVpeak = FV;
    %FV = [kuyruk(:); nosem-nosebefore];  nosebefore = nosem;  FVpeak = FV;
    %FV = FV - mean(FV);  FV = FV./std(F);
%    FV = WLDA'*FV;
%    happy = svmclassify(SVMStructHappy, FV');
%    thinking = svmclassify(SVMStructThinking, FV');
%    agree = svmclassify(SVMStructAgree, FV');
%    disagree = svmclassify(SVMStructDisagree, FV');
%    anger = svmclassify(SVMStructAnger, FV');
%    suprise = svmclassify(SVMStructSuprise, FV');
    %normal = svmclassify(SVMStructNormal, FV'); 
    
 %   karar = [happy, thinking,agree,disagree,anger,suprise];
    %if ~any(karar)
    %    karar = 7;
    %end
    %karar = decide_me([happy,thinking,agree,disagree,anger,suprise], peakV, FVpeak);
    
    karar = nnet(FV);  
    [~,karar] = max(karar);
    %karar = predict(tree,FV');
  
    if farkt < (fark*16.5)/19 && ~up && ~down
        karar = 5;
    end
    if leftC >= 1 || rightC >= 1
        karar = 4;
    end
    if upC >= 1 || downC >=1
        karar = 3;
    end
    if farklipst >= (farklipsson*5.3)/4 && ~left && ~right
        karar = 1;
    end
    if farkt >= fark + 1 && farklips2t >= (farklips2son*2)/3
        karar = 6;
    end
    %if solCC>=1 || sagCC>=1
    %    karar = 2;
    %end
    kararV(karar) = kararV(karar) + 1;
    if sayac == tekrar
        sayac = 0;
        [~,ind] = max(kararV);
        %if ind == 4
        %    aaa = 1;  %disp(iter);
        %end
        fprintf('%s\n',S{ind});
        %imshow(uint8(image));  drawnow;
        kararV = zeros(1,7);
    end
    sayac = sayac + 1;
    iter = iter + 1; %disp(iter);
    image = getsnapshot(vidobj);
    image = imfilter(image, hgauss);
    image = imresize(image, scaleC);
    farklips2 = farklips2 + farklips2t;
    farklips = farklips + farklipst;
    if iter == 5
        farklips2son = floor(farklips2/5);
        farklipsson = floor(farklips/5);
    end
    %if iter == 242
    %    aaa = 1;
    %end
end    
    

stop(vidobj);
delete(vidobj);

end

function [up,down,right,left] = findYon(xmb,ymb, xmn, ymn)
up = false;
down = false;
right = false;
left = false;
switch sign(xmn-xmb)
    case 1,
        if xmn -xmb >= 2
            down = true;
        end
    case -1,
        if xmb-xmn >= 2
            up = true;
        end
end
switch sign(ymn-ymb)
    case 1,
        if ymn-ymb >= 2
            right = true;
        end
    case -1,
        if ymb - ymn >= 2
            left = true;
        end
end
end

function karar = decide_me(States,peakV,Fpeak)
if ~any(States)
    karar = 7;
    return;
end
fark = peakV(:,States) - repmat(Fpeak,1,sum(States));
fark = sum(fark.^2);
[~,ind] = min(fark);
S = find(States);  karar = S(ind);
end
