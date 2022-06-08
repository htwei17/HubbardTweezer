function y = Cin(x)
%Return the value of the modified cosine integral Cin(x)=\gamma+log x-Ci(x), with \gamma the Euler-Mascheroni constant.
%This routine uses the facts that Cin(-x)=Cin(x) and Cin(0)=0
%Used to define the integrals of the sinc DVR basis functions over the interval [-\infty,0]

xin=x;
if x<0.0
    xin=-x;
end
if norm(x)<1e-12
    y=0.0;
else
    y=vpa(eulergamma)+log(xin)-cosint(xin);
end
end
