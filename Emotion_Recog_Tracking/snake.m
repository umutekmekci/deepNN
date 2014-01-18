function [ coords ] = snake(imH, coords, negP, negS,W, E_old)

[dx,dy] = meshgrid(-negS:negS,-negS:negS);   %state coordinates
D = [dx(:)';dy(:)'];    
fark2 = zeros(size(D,2));
for i = 1:size(D,2)
    fark2(:,i) = sum((D - repmat(D(:,i),1,size(D,2))).^2)';
end
fark2 = mapstd(fark2);
fark1 = sum(abs(D))';
fark1 = (fark1-mean(fark1))/std(fark1);
fark1 = fark1*0.08;
fark2 = fark2*0.08;

numState = (2*negS+1)*(2*negS+1);  %number of state how many states a variable can be
numVar = size(coords,2);           % number of random variables
probTable = ones(numState)/(numState.^2);  % transition probability
E = zeros(numState, numVar);       % min energy functions for each variable and state
sira = numVar:-1:1;
siraLogic = true;

maxIter = 100;
iter = 1;
while iter < maxIter
    Et = energyFunc(imH,coords(:,1),D, negP,W);    % energy function for variable1
    Et = sum( (Et-repmat(E_old(:,1),1,size(Et,2))).^2 )';
    E(:,1) = (Et-mean(Et))/std(Et);
    E(:,1) = E(:,1) + fark1;
    Mvec = zeros(numState, numVar-1);         % Maximum Vector for back tracking
    for i = 2:numVar
        Et = energyFunc(imH, coords(:,i), D, negP, W);
        Et = sum( (Et-repmat(E_old(:,i),1,size(Et,2))).^2 )';
        Et = (Et-mean(Et))/std(Et);
        [M, ind] = min( (repmat(E(:,i-1),1,numState).*probTable) + fark2 );
        E(:,i) = M' + Et + fark1;
        Mvec(:,i-1) = ind';
    end
    states = zeros(1,numVar);
    [~,states(end)] = min(E(:,end));
    for i = size(Mvec,2):-1:1
        states(i) = Mvec(states(i+1),i);
    end

    new_coords = coords + D(:,states);
    if all((all(coords == new_coords)))
        if ~siraLogic
            coords = coords(:,sira);
        end
        break;
    end
    coords = new_coords;
    siraLogic = ~siraLogic;
    coords = coords(:,sira);
    iter = iter + 1;
end

end