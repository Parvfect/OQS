%top programme
qxs=[0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.425 0.45 0.475 0.48 0.485 0.49 0.495 0.4975 0.4995 0.49995 0.5 0.50005 0.5005 0.5025 0.505 0.51 0.515 0.52 0.525 0.55 0.575 0.6 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1];
n=40;
X=zeros(n);A=X;AP=X;SQ=X;H=X;H0=X;
%constants
hv=0.67*9.99*10.ˆ−22;h=1.0547*10.ˆ−34;v=hv/h;
C=5*10.ˆ−15;L=3*10.ˆ−10;w=1/sqrt(L*C);hw=h*w;
phi0=2.067*10.ˆ−15;
a=hv/hw;
kQ=2*pi*sqrt(hw*L)/phi0;
g=0.55;
W=25*w;
beta= 4*pi.ˆ2*hv*L/(phi0.ˆ2);
goverw=0.005;%qx is really qx/phi0
% annihilation operator
for j=1:(n−1),
A(j,j+1)=sqrt(j);
end
%other operators
AP=A';
X=(AP+A)/sqrt(2);
P=(A−AP)/(sqrt(2)*1i)
ID=eye(n);
H0=AP*A+1/2*ID;
A1=0.9348;
x1=0.4464;
p1=−0.07082−0.8887i;
j1=0.07002−0.03285i;
A2=4.79;
x2=0.6321;
p2=0.005154+0.371i;
j2=0.6802−0.009269i;
load expvals120 e;
e=e(1:40,1:40);
vals=[];
for k=1:length(qxs),
k
qx=qxs(k)
e1=e*exp(1i*2*pi*qx);CosQ=real(e1); SinQ=imag(e1);
HXP=goverw*(3*gˆ2/2−g+1/2)*(X*P+P*X);
HXJ=goverw*w/(2*W)*sqrt(beta*v/w)*X*SinQ;
HPJ= goverw*gˆ2/2*(sqrt(beta*v/w)*(P*SinQ+SinQ*P)+beta*w/(2*W)*CosQ);
%HPJ=0;
H=H0+HXP−HXJ−HPJ−a*CosQ;
%Lindblad
L1=sqrt(A1*goverw)* (x1*X +p1*P+ j1*SinQ);
LP1=L1';
L2=sqrt(A2*goverw)*(x2*X+ p2*P +j2*SinQ);
LP2=L2';
%L = sqrt(goverw)*A;
%LP=L';
%Solve AMp + pBM+ CMpDM=0
%0=−i[H,p]+LpLP−0.5LPLp−0.5pLLP
AM=−1i*H−LP1*L1/2−LP2*L2/2;
BM=+1i*H−LP1*L1/2−LP2*L2/2;
CM=L1;
DM=LP1;
EM=L2;
FM=LP2
G=kron(ID,AM)+kron(transpose(BM),ID)...
+kron(transpose(DM),CM)+kron(transpose(FM),EM);
Solve G*vec[rho]=0
rhonull=null(G);
rankG=rank(G)%should be nˆ2−1
M=G(1:nˆ2−1,1:nˆ2−1);
rankM=rank(M)%should still be nˆ2−1
v1=zeros(1,nˆ2−1);
v1(1:n+1:nˆ2−1)=1;
v2(1:nˆ2−1)=G(1:nˆ2−1,nˆ2);
clear G
M1=transpose(v2)*v1;
M2=M−M1;clear M M1 rho
rho=−M2\transpose(v2);rho=[rho;0];%extend rho to nˆ2 terms
clear M2
rho=reshape(rho,n,n);%turn square
rho(n,n)=1−trace(rho);
[UU,DD]=eig(rho);
evals=diag(DD);
evals(1:6)
purity(k)=trace(rho*rho)
vals=[vals,[purity(k);qxs(k)]]
Z=AM*rho + rho*BM+ CM*rho*DM +EM*rho*FM;%check that everything works out
ck(k)=max(max(abs(Z)));
ks=num2str(k);
str=['rho g055 W25w n40 ',ks];
save(str, 'rho', 'Z')%, 'rhonull')
end
purityg055W25wn40 = purity;
ave('purityg055W25wn40', 'purityg055W25wn40')
figure
plot(qxs,purity,'kd−');
axis([0 1 0 1])
xlabel('external flux, $$\Phi x/\Phi 0$$','interpreter', 'latex');
ylabel('steady state purity, Tr$$(\rhoˆ2)$$','interpreter', 'latex')
title('g=0.55, $$\Omega=25\omega 0$$','interpreter', 'latex')