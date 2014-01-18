function [ altd, sagd, sold, nose ] = imageAnalyzeMouth( oimage,altd,sagd, sold, nose, ley, rey, neg, im )

%imshow(uint8(im)); hold on;
bbv = {altd, sold, sagd, nose};
sel = 1;
cikis = [false,false,false,false];

iter = 0;
while iter < 100
   
    bb1 = bbv{sel};
    
    %bb=boundingBox([bb1(1,1),bb1(1,2);bb1(2,1),bb1(2,2)]);
    %plot(bb(2,:),bb(1,:),'b.'); drawnow;
    [yc,xc]=meshgrid(-neg:neg,-neg:neg);
    C = [xc(:)';yc(:)']*2;

    x1 = bb1(1,1);  x2 = bb1(1,2);
    y1 = bb1(2,1);  y2 = bb1(2,2);
    
    mask = getMask(sel, x1,x2,y1,y2);
    
    iter = 0;
    P = zeros(1,size(C,2));
    onlycount = true(1,size(C,2));
    %toplam = 255*numel(mask);
    for i = 1:size(C,2)
        nc = [x1,x2;y1,y2] + repmat(C(:,i),1,2);
        nc(nc<0) = 1;
        onlycount(i) = getSinir(nc, bbv,ley,rey, sel);
        R = im(nc(1,1):nc(1,2),nc(2,1):nc(2,2),1);
        %B = im(nc(1,1):nc(1,2),nc(2,1):nc(2,2),3);
        switch sel
            case 1, %|| sel ==4
                P(i) = sum(sum((oimage(nc(1,1):nc(1,2),nc(2,1):nc(2,2),1)-oimage(nc(1,1):nc(1,2),nc(2,1):nc(2,2),2)).*mask)) + sum(R(:).*(-mask(:))); %+ (toplam - sum(sum(abs(diff(image(nc(1,1):nc(1,2),nc(2,1):nc(2,2),1))))));
            case 4,
                 %P(i) = sum(sum((oimage(nc(1,1):nc(1,2),nc(2,1):nc(2,2),1).*mask)));
                
                P(i) = sum(sum((R).*mask)) - sum(sum(R));  %+ (toplam - sum(sum(abs(diff(R)))));
            otherwise,
                %mask = -mask;
                %P(i) = sum(sum((oimage(nc(1,1):nc(1,2),nc(2,1):nc(2,2),1)-oimage(nc(1,1):nc(1,2),nc(2,1):nc(2,2),2)).*mask))+ sum(R(:).*(-mask(:)));
                %P(i) = sum(R(:).*mask(:));
                P(i) = sum(sum((oimage(nc(1,1):nc(1,2),nc(2,1):nc(2,2),1).*mask)));
                %P(i) = sum(sum(( ( im(nc(1,1):nc(1,2),nc(2,1):nc(2,2),3) - im(nc(1,1):nc(1,2),nc(2,1):nc(2,2),1) ).*-mask)));
                %temp_r = im(nc(1,1):nc(1,2),nc(2,1):nc(2,2),1);  %temp_b = im(nc(1,1):nc(1,2),nc(2,1):nc(2,2),3); 
                %P(i) = sum(sum(temp_r.*mask)) + sum(sum((oimage(nc(1,1):nc(1,2),nc(2,1):nc(2,2),1).*mask))); %+ sum(temp_b(mask == -1)));
        end
    end
    indV = 1:size(C,2);
    indV = indV(onlycount);
    P = P(onlycount);
    [~,ind] = max(P);
    ind = indV(ind);
    if ~any(onlycount)
        ind = floor(length(onlycount)/2)+1;
    end
    if ind == (floor(length(onlycount)/2)+1)
        cikis(sel) = true;
    else
        cikis(sel) = false;
    end
    
    x1 = x1 + C(1,ind);  x2 = x2 + C(1,ind);
    y1 = y1 + C(2,ind);  y2 = y2 + C(2,ind);
    iter = iter + 1;

    bbv{sel} = [x1,x2;y1,y2];
    sel = max(mod(sel+1,5),1);
    
    if all(cikis)
        break;
    end
    
end

masksol = getMask(2, sold(1,1),sold(1,2),sold(2,1),sold(2,2));
masksag = getMask(3, sagd(1,1),sagd(1,2),sagd(2,1),sagd(2,2));
altd = bbv{1};  sold = bbv{2};
sagd = bbv{3};  %ustd = bbv{4};
nose = bbv{4};
P1 = sum(sum((oimage(sold(1,1):sold(1,2),sold(2,1):sold(2,2),1).*masksol))) + sum(sum((oimage(sagd(1,1):sagd(1,2),sagd(2,1):sagd(2,2),1).*masksag)));

tsold = sold;  tsold(1,:) = sagd(1,:);
P2 = sum(sum((oimage(tsold(1,1):tsold(1,2),tsold(2,1):tsold(2,2),1).*masksol))) + sum(sum((oimage(sagd(1,1):sagd(1,2),sagd(2,1):sagd(2,2),1).*masksag)));

tsagd = sagd;  tsagd(1,:) = sold(1,:);
P3 = sum(sum((oimage(sold(1,1):sold(1,2),sold(2,1):sold(2,2),1).*masksol))) + sum(sum((oimage(tsagd(1,1):tsagd(1,2),tsagd(2,1):tsagd(2,2),1).*masksag)));

[~,ind] = max([P1,P2,P3]);
switch ind
    case 2,
        sold = tsold;
    case 3,
        sagd = tsagd;
end

%nose_m = floor((nose(2,1)+nose(2,2))/2);
%altd_m = floor((altd(2,1)+altd(2,2))/2);
%fark = nose_m-altd_m;
%altd(2,:) = altd(2,:) + [fark,fark];

end

function onlycount = getSinir(nc, bbv,ley,rey, sel)
onlycount = true;
alt = bbv{1};  sol = bbv{2};  sag = bbv{3};   nose = bbv{4}; %ust = bbv{4};
ncx1 = nc(1,1);  ncx2 = nc(1,2);  ncy1 = nc(2,1);  ncy2 = nc(2,2);
ncym = floor((ncy1 + ncy2)/2);
altx1 = alt(1,1);  altx2 = alt(1,2);  alty1 = alt(2,1);  alty2 = alt(2,2);
solx1 = sol(1,1);  solx2 = sol(1,2);  soly1 = sol(2,1);  soly2 = sol(2,2);
solxm = floor((solx1 + solx2)/2);
sagx1 = sag(1,1);  sagx2 = sag(1,2);  sagy1 = sag(2,1);  sagy2 = sag(2,2);
sagxm = floor((sagx1 + sagx2)/2);
nosex1 = nose(1,1);  nosex2 = nose(1,2);  nosey1 = nose(2,1);  nosey2 = nose(2,2);
nosexm = floor((nosex1 + nosex2)/2);
leyy1 = rey(2,1);  leyy2 = rey(2,2);  leym = floor((leyy1 + leyy2)/2);  
reyy2 = ley(2,2);  reyy1 = ley(2,1);  reym = floor((reyy1 + reyy2)/2);
%ustx1 = ust(1,1);  ustx2 = ust(1,2);  usty1 = ust(2,1);  usty2 = ust(2,2);
switch sel
    case 1, 
        if (ncy1<soly2 || ncy2 > sagy1 || ncx1 < solx1 || ncx1 < sagx1) % || ncx1 < ustx2)
            onlycount = false;
        end
    case 2,
        if ncy2 > alty1 || ncx1 > altx1 || ncx1 < nosexm || ncy1 < leyy1 || ncy2 > leyy2 %|| ncy2 > usty1
            onlycount = false;
        end
    case 3,
        if ncy1 < alty2 || ncx1 > altx1 || ncx1 < nosexm || ncy2 > reyy2 || ncy1 < reyy1 %|| ncy1 < usty2
            onlycount = false;
        end
    case 4,
        if ncx2 > solxm || ncx2 > sagxm || ncx1 < ley(1,2) || ncx1 < rey(1,2)
            onlycount = false;
        end
    %case 4,
    %    if ncy1 < soly2 || ncy2 > sagy1 || ncx2 > altx1
    %        onlycount = false;
    %    end
end
end


function mask = getMask(sel,x1,x2,y1,y2)

mask = ones(x2-x1+1,y2-y1+1);

if sel == 4
    gen = y2-y1+1;
    yuk = x2-x1+1;
    xfark = floor(yuk/5);
    yfark = floor(gen/10);
    X = floor(0.5*xfark):floor(4.5*xfark);  X(X == 0) = 1;
    Y1 = 2*yfark:floor(4.5*yfark);
    Y2 = floor(5.5*yfark):8*yfark;
    mask(X,Y1) = -1;
    mask(X,Y2) = -1;
    return;
end


if sel == 1
    mask(floor((x2-x1)/2)+1:end,:) = -1;
    return
end

step = floor((x2-x1+1)/5)+1;


if sel == 2
    mask(step:end-step,floor((y2-y1)/2)-2:end) = -1;
    %mask = triu(mask);  mask(mask == 0) = -1;  mask = -mask;
    %mask(:,floor((y2-y1)/2)+1:end) = -1;
else 
    mask(step:end-step,1:floor((y2-y1)/2)+2) = -1;
    %mask = triu(mask); mask(mask == 0) = -1;  mask = fliplr(mask);  mask = -mask;
    %mask(:,1:floor((y2-y1)/2)) = -1;
end
     


%mask = -ones(x2-x1+1,y2-y1+1);
%[sat,sut] = size(mask);
%if sel == 4
%    mask(floor(sat/2):end,:) = 1;
%    return;
%end

%mask(:,floor(sut/2)+1:end) = 1;
%if sel == 3
%    return;
%else
%    mask = -mask;
%end

end

