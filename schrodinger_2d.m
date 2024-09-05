dom = [-5,5,-5,5];
d = 300; % Discretization size
nx = d;
ny = d;
x_sample = linspace(dom(1),dom(2),nx)';
y_sample = linspace(dom(1),dom(2),ny)';

% Compute Wx quadrature weight matrix
dx = diff(x_sample);
Wx = zeros(nx,1);
Wx(2:end-1) = dx(2:end)+dx(1:end-1);
Wx(1) = dx(1);
Wx(end) = dx(end);
Wx = Wx/2;
Wx = spdiags(Wx, 0, nx, nx);

% Compute Wy quadrature weight matrix
dy = diff(y_sample);
Wy = zeros(ny,1);
Wy(2:end-1) = dy(2:end)+dy(1:end-1);
Wy(1) = dy(1);
Wy(end) = dy(end);
Wy = Wy/2;
Wy = spdiags(Wy, 0, ny, ny);

% Compute quadrature weight W
W = kron(Wy,Wx);

% Generate evaluation grid
[X_sample,Y_sample] = meshgrid(x_sample, y_sample);

% Harmonic potential
V = chebfun2(@(x,y)0.5*(x.^2+y.^2),dom);

% Number of initial conditions on each direction
Ninit = 20; 
x_init = linspace(-4,4,Ninit);
[x_gauss, y_gauss] = meshgrid(x_init,x_init);
x_gauss = x_gauss(:); y_gauss = y_gauss(:);
N = length(x_gauss);

PX = zeros(nx*ny,N);
PY = zeros(nx*ny,N);
for n = 1:N
    sprintf("n = %d / %d",n, N)
    u_init = chebfun2(@(x,y)(1+1i)*exp(-((x-x_gauss(n)).^2+(y-y_gauss(n)).^2)*sqrt(N)/5),dom);
    u = -0.5*(diff(u_init,2,1)+diff(u_init,2,2)) + V.*u_init;
    PX(:,n) = u_init(X_sample(:),Y_sample(:));
    PY(:,n) = u(X_sample(:),Y_sample(:));
end

%% Compute eigenvalues and eigenfunctions

% Compute Gram matrix
G = (PX'*W*PX);

% Compute solution to Hermitian DMD
K = G \ (PX'*W*PY+PY'*W*PX)/2;

% Compute eigenvalues and eigenvectors
[V,E] = eig(K,'vector');
[E,I] = sort(E,'ascend','ComparisonMethod','abs');
V = V(:,I);
nn = real(dot(V,G*V));
V = (1./sqrt(nn)).*V;
E = real(E);

%% Plot eigenmodes
for i = 1:6
    subplot(3,3,i)
    surf(X_sample,Y_sample,reshape(abs(PX*V(:,i)).^2, length(y_sample), length(x_sample)))
    view(0,90)
    xlabel("$x$",Interpreter="latex")
    ylabel("$y$",Interpreter="latex")
    title(sprintf("$\\lambda_j = %.1f$",E(i)),Interpreter="latex")
    colormap jet
    clim(1) = 0;
    colorbar
    shading interp
    axis square
end

%% Plot computed eigenvalues
plot(real(E),'.')
E_exact = (0:10)+(0:10)'+1;
hold on
plot(sort(E_exact(:)),'o')
xlim([0,100])
hold off
axis square
shg

%% Compute cj coefficient in expansion mu_v = sum |vj^*v|^2 delta_{lambda_j}

% Create function f
f = @(x,y)sin(pi*x/5).*sin(pi*y/5);

% Sample the function
F = f(X_sample(:),Y_sample(:));
cj = abs((PX*V)'*W*F).^2;

% Get exact eigenvalues
E_exact = [];
for ix = 0:30
    for iy = 0:30
        if ix+iy+1 <= 30
            E_exact = [E_exact, ix+iy+1];
        end
    end
end
E_exact = sort(E_exact,'ascend')';

% Sum for same value of eigenvalues
E_exact = E_exact(1:numel(cj));
G = findgroups(E_exact);
Cj = accumarray(G,cj);
E_unique = accumarray(G,E,size(Cj),@mean);

close all
for idx = 1 : numel(Cj)
    plot([E_unique(idx), E_unique(idx)], [0, Cj(idx)],'blue',LineWidth=2);
    hold on;
end
hold off
xlim([0,20])
xlabel("$\lambda_j$",Interpreter="latex")
ylabel("$c_j$",Interpreter="latex")
title(sprintf("Approximation at $M = %d^2$",d),Interpreter="latex")

%% Do the same for exact eigenvalues
n_max = 29;
cj_exact = [];
E_exact = [];
G = exp(-(X_sample(:).^2+Y_sample(:).^2)/2);
H = hermpoly(0:n_max);
Hx = H(X_sample(:),:);
Hy = H(Y_sample(:),:);
for ix = 0:n_max
    sprintf("ix = %d / %d",ix,n_max)
    for iy = 0:n_max
        if ix+iy+1 <= 30
            U = Hx(:,ix+1).*Hy(:,iy+1).*G;
            U = U / sqrt(sum(W*U.^2));
            cj_exact = [cj_exact,abs(U'*W*F).^2];
            E_exact = [E_exact, ix+iy+1];
        end

    end
end
[~,i_sort] = sort(E_exact,'ascend');
E_exact = E_exact(i_sort)';
E_exact = E_exact(1:numel(cj));
cj_exact = cj_exact(i_sort)';
cj_exact = cj_exact(1:numel(cj));
G = findgroups(E_exact);
Cj_exact = accumarray(G,cj_exact);
cj_exact = Cj_exact(G);
E_exact_unique = unique(E_exact);

close all
for idx = 1 : numel(Cj_exact)
    plot([E_exact_unique(idx), E_exact_unique(idx)], [0, Cj_exact(idx)],'blue',LineWidth=2);
    hold on;
end
hold off
xlim([0,20])
ylim(1) = 0;
xlabel("$\lambda_j$",Interpreter="latex")
ylabel("$c_j$",Interpreter="latex")
