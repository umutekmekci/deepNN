function [ bb1,bb2,bb3,bb4 ] = imageAnalyzeGenel( image,bb1,bb2,bb3,bb4,neg1,neg2 )

%imshow(uint8(image)); hold on;
bbv1 = {bb1,bb3};
bbv2 = {bb2,bb4};
bbvv = {bbv1,bbv2};
negv = {neg1,neg2};
sel = [true,false];
cikis = [false,false];
whichbb = 1;

iter = 0;
while iter < 100
   
    bbv = bbvv{whichbb};
    bb1 = bbv{sel};
    neg = negv{sel};
    
    %bb=boundingBox([bb1(1,1),bb1(1,2);bb1(2,1),bb1(2,2)]);
    %plot(bb(2,:),bb(1,:),'b.'); drawnow;
    [yc,xc]=meshgrid(-neg:neg,-neg:neg);
    C = [xc(:)';yc(:)']*2;

    x1 = bb1(1,1);  x2 = bb1(1,2);
    y1 = bb1(2,1);  y2 = bb1(2,2);

    gen = y2-y1;
    yuk = x2-x1;
    xfark = ceil(yuk/5);
    yfark = ceil(gen/7);
    %x11 = x1 + xfark;  x22 = x2-xfark;
    %y11 = y1 + yfark;  y22 = y2-yfark;

    %mask = 1;
    if sel(2)
        mask = -ones(x2-x1+1,y2-y1+1);
        mask(xfark:yuk-xfark+1,yfark:gen-yfark+1) = 1;
    else
         mask = ones(x2-x1+1,y2-y1+1);
         mask(1:floor((x2-x1)/2 + 1),:) = -1;
    end

    iter = 0;
    P = zeros(1,size(C,2));
    sinirbb = bbv{~sel};
    onlycount = true(1,size(C,2));
    toplam = 255*numel(mask);
    for i = 1:size(C,2)
        nc = [x1,x2;y1,y2] + repmat(C(:,i),1,2);
        if sel(1)
            if nc(1,2) > sinirbb(1,1)
                onlycount(i) = false;
            end
        end
        if sel(2)
            if nc(1,1) < sinirbb(1,2)
                onlycount(i) = false;
            end
        end
        P(i) = sum(sum(image(nc(1,1):nc(1,2),nc(2,1):nc(2,2),1).*mask)) + (toplam - sum(sum(abs(diff(image(nc(1,1):nc(1,2),nc(2,1):nc(2,2),1))))));
    end
    indV = 1:size(C,2);
    indV = indV(onlycount);
    P = P(onlycount);
    [~,ind] = min(P);
    ind = indV(ind);
    if ind == (floor(length(onlycount)/2)+1)
        cikis(sel) = true;
    else
        cikis(sel) = false;
    end
    
    x1 = x1 + C(1,ind);  x2 = x2 + C(1,ind);
    y1 = y1 + C(2,ind);  y2 = y2 + C(2,ind);
    iter = iter + 1;

    bbvv{whichbb}{sel} = [x1,x2;y1,y2];
    sel = ~sel;
    
    if all(cikis)
        if whichbb == 2
            break;
        else
            whichbb = 2;
        end
    end
    
end

bb1 = bbvv{1}{1};  bb2 = bbvv{2}{1};
bb3 = bbvv{1}{2};  bb4 = bbvv{2}{2};

end

