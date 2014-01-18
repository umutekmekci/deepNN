function playVideo_test

name='sonuclar.mat';
matobj = matfile(name);
labels_test = matobj.labels_test;
frame_test = matobj.frame_test;
coords_test = matobj.coords_test;
svm_test = matobj.svm_test;
names = matobj.names;
clear matobj


[frame_test, ind] = sort(frame_test);
labels_test = labels_test(ind);
svm_test = svm_test(ind);
coords_test = coords_test(ind,:);

frame_test_t = frame_test;
labels_test_t = labels_test;
svm_test_t = svm_test;
coords_test_t = coords_test;

videoPlayer  = vision.VideoPlayer('Position',[200 0 1024 1024]);
vidObj = VideoReader('The.Big.Bang.Theory.S01E01.HDTV.XviD-XOR.avi');

ind = svm_test ~= labels_test;
frame_test = frame_test(ind);
labels_test = labels_test(ind);
svm_test = svm_test(ind);
coords_test = coords_test(ind,:);
frame_test = frame_test(1:20);
labels_test = labels_test(1:20);
svm_test = svm_test(1:20);
coords_test = coords_test(1:20,:);


frames_to_show = [];
for i = 1:length(frame_test)
    ff = frame_test(i);
    bas = ff-10;  son = ff+10;
    frames_to_show = [frames_to_show, bas:son];
end
%frames_to_show = unique(frames_to_show);

green_rec=vision.ShapeInserter('BorderColor','Custom','CustomBorderColor',uint8([0 255 0]));
red_rec=vision.ShapeInserter('BorderColor','Custom','CustomBorderColor',uint8([255 0 0]));
textInserter = vision.TextInserter('FontSize', 24);

kk = 1;
sayac = 1;
for i=1:length(frames_to_show)
    image=read(vidObj,frames_to_show(i));
    image=imresize(image,[576 1024]);
    if kk <= length(frame_test)
        tt = frame_test(kk);
    end
    if frames_to_show(i) == tt
        tag=cell2mat(names(svm_test(kk)));
        tag = [tag, ' ', num2str(frames_to_show(i))];
        sayac = sayac + 1;
        release(textInserter)
        set(textInserter, 'Text', tag, 'Color',uint8([255 0 0]), 'Location', int32(coords_test(kk,1:2)) )
        image=step(red_rec,image,int32(coords_test(kk,:)));
        image=step(textInserter,image);
        kk = kk + 1;        
        step(videoPlayer, image);
        pause(1)
    else
        step(videoPlayer, image);
    end
end

ind = svm_test_t == labels_test_t;
frame_test = frame_test_t(ind);
labels_test = labels_test_t(ind);
svm_test = svm_test_t(ind);
coords_test = coords_test_t(ind,:);
frame_test = frame_test(300:320);
labels_test = labels_test(300:320);
svm_test = svm_test(300:320);
coords_test = coords_test(300:320,:);


frames_to_show = [];
for i = 1:length(frame_test)
    ff = frame_test(i);
    bas = ff-10;  son = ff+10;
    frames_to_show = [frames_to_show, bas:son];
end

kk = 1;
sayac = 1;
for i=1:length(frames_to_show)
    image=read(vidObj,frames_to_show(i));
    image=imresize(image,[576 1024]);
    if kk <= length(frame_test)
        tt = frame_test(kk);
    end
    if frames_to_show(i) == tt
        tag=cell2mat(names(svm_test(kk)));
        tag = [tag, ' ', num2str(frames_to_show(i))];
        sayac = sayac + 1;
        release(textInserter)
        set(textInserter, 'Text', tag, 'Color',uint8([0 255 0]), 'Location', int32(coords_test(kk,1:2)) )
        image=step(green_rec,image,int32(coords_test(kk,:)));
        image=step(textInserter,image);
        kk = kk + 1;        
        step(videoPlayer, image);
        pause(1)
    else
        step(videoPlayer, image);
    end
end




release(videoPlayer);

end

