function [A, D]=GetDWT(x,N,wname)
        

    [C, L]=wavedec(x,N,wname);

    A=cell(N,1);
    D=cell(N,1);
    for i=1:N
        A{i}=wrcoef('a',C,L,wname,i);
        D{i}=wrcoef('d',C,L,wname,i);
    end

end