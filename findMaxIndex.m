function class_i = findMaxIndex(autoEncoded)
class=ones(size(autoEncoded,2),1)*(-10);
class_i=ones(size(autoEncoded,2),1)*(-10);

for i=1:size(autoEncoded,2)
    for j=1:size(autoEncoded,1)
        if class(i)<autoEncoded(j,i)
            class(i)=autoEncoded(j,i);
            class_i(i)=j;
        end
    end
end

end