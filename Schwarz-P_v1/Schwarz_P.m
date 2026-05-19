clc;
clear;
close all;

to_mat = true; %Option to save mesh data or not
draw_graph = true; %Option to display the mesh in MATLAB

Size  = 2e-3; %TPMS cell edge dimensions, initially generates unit cell then scale
ds = 0.001; %Smallest step size, relative to the unit cell

%Fraction of nodes to be used, control over resolution
frac = 25;
Ds = frac*ds;

%Expansion layer sizing controls
lvl_init = 0; %Surface isofunction value
mesh_dir = 0; %Direction in which the mesh is built: inside (0) or outside (1) the first isolayer
l1_ht = Size*Ds/10; %First layer height
exp_ratio = 1.05; %Ratio of subsequent layer heights
n_layers = 0;
%Set n_Layers = 0 to control the total inflation region thickness instead
BL_ht = 25*l1_ht; %Ignored if n_layers is non-zero

%Control for mesh sizing along the unit cell edge parallel to the flow
%direction (assumed to be y)
factor_streamwise = 1; %Parameter not used in MATLAB but passed to Python

%%
%Creates starting points at prescribed isolevel on three orthagonal unit
%cell faces
P = zeros(1,3);
P(1,1) = 0.5;
P = find_level(P,lvl_init,ds,[0,1,0]);
radius = norm(P-[0.5,0,0]);
n1 = floor(pi*radius/(2*Ds))+1;
n1 = n1 + mod(4-mod(n1,4),4);
theta = (-pi/4:pi/(2*(n1+1)):pi/4);
theta = theta(2:end-1);

n_points = 1000;
x = zeros(n_points+1,2*n1);
y = zeros(n_points+1,2*n1);
z = zeros(n_points+1,2*n1);

for i = 1:n1
    dir = unit([0,cos(theta(i)),sin(theta(i))]);
    P = [0.5,radius*dir(2),radius*dir(3)];
    P = find_level(P,lvl_init,ds,dir);
    x(1,i) = P(1,1); y(1,i) = P(1,2); z(1,i) = P(1,3);
    z(1,n1+i) = z(1,i); y(1,n1+i) = x(1,i); x(1,n1+i) = y(1,i);
end

%%
%Drawing contour lines from the edge of unit cell inwards
count_all = (n_points+1)*ones(1,width(x));
for i = 1:width(x)
    p_array = [x(1,i);y(1,i);z(1,i)];
    [~,I] = max(abs(p_array));
    dir = zeros(1,3);
    dir(1,I) = -sign(p_array(I));
    for j = (2:n_points+1)
        curv_vect = principal_directions([x(j-1,i),y(j-1,i),z(j-1,i)]);
        [~,I] = max(abs(dir*curv_vect));
        dir = sign(dir*curv_vect(:,I))*curv_vect(:,I)';
        P = next_point_vector([x(j-1,i),y(j-1,i),z(j-1,i)],lvl_init,ds,dir);
        x(j,i) = P(1); y(j,i) = P(2); z(j,i) = P(3);
        dir = unit([x(j,i)-x(j-1,i),y(j,i)-y(j-1,i),z(j,i)-z(j-1,i)]);
        if max([abs(x(j,i)),abs(y(j,i)),abs(z(j,i))]) > 0.5
            count_all(i) = j-1;
            x(j:end,i) = x(j-1,i); y(j:end,i) = y(j-1,i); z(j:end,i) = z(j-1,i);
            break
        end
    end
end
clear P curv_vect dir radius theta

%%
%Section for merging opposite lines
for i = 1:n1
    I = n1+i;
    old_count = count_all(i);
    count_all(i) = max(count_all(i),count_all(I));
    count_all(I) = count_all(i);
    p_new = zeros(n_points+1,3);
    for j = 1:count_all(i)
        frc = (j-1)/(count_all(i)-1);
        j_ = [floor(frc*(old_count-1))+1,floor(frc*(old_count-1))+2,...
            floor((1-frc)*(count_all(I)-1))+1,floor((1-frc)*(count_all(I)-1))+2];
        a = [j_(2)-(frc*(old_count-1)+1),(frc*(old_count-1)+1)-j_(1),...
            j_(4)-((1-frc)*(count_all(I)-1)+1),((1-frc)*(count_all(I)-1))+1-j_(3)];
        j_(2) = min(max(j_(2),1),old_count);
        j_(4) = min(max(j_(4),1),count_all(I));
        p_new(j,:) = (1-frc)*(a(1)*[x(j_(1),i),y(j_(1),i),z(j_(1),i)]...
            + a(2)*[x(j_(2),i),y(j_(2),i),z(j_(2),i)])...
            + frc*(a(3)*[x(j_(3),I),y(j_(3),I),z(j_(3),I)]...
            + a(4)*[x(j_(4),I),y(j_(4),I),z(j_(4),I)]);
        p_new(j,:) = grad_move(p_new(j,:),lvl_init);
    end
    p_new(count_all(i):end,:) = ones(n_points+2-count_all(i),1)*p_new(count_all(i),:);
    x(:,i) = p_new(:,1); y(:,i) = p_new(:,2); z(:,i) = p_new(:,3);
    x(1:count_all(I),I) = x(count_all(i):-1:1,i);
    x(count_all(I):end,I) = x(count_all(I),I);
    y(1:count_all(I),I) = y(count_all(i):-1:1,i);
    y(count_all(I):end,I) = y(count_all(I),I);
    z(1:count_all(I),I) = z(count_all(i):-1:1,i);
    z(count_all(I):end,I) = z(count_all(I),I);
end

%%
n2 = floor(sqrt(max(count_all(1:n1))*min(count_all(1:n1)))/frac)+1;
n2 = n2 + mod(n2,2); %n2 should be even for a single point at the 'stagnation zone' 
indices = zeros(n2+1,2*n1); %Assigned numerical indices of nodes
p_array = zeros(n1*(n2+1),3); %List of all nodes as per their assigned index
for i = 1:n1
    for k = 1:n2+1
        i_ = 1 + round((count_all(i)-1)*(k-1)/n2);
        indices(k,i) = (n2+1)*(i-1)+k;
        indices(n2+2-k,n1+i) = indices(k,i);
        p_array((n2+1)*(i-1)+k,:) = [x(i_,i),y(i_,i),z(i_,i)];
    end
end

%Imposing symmetry in x=y plane
a = (1:n1*(n2+1));
a = (a+n2-2*mod(a-1,n2+1)).';
p_array = (p_array + p_array(a,:)*[0,1,0;1,0,0;0,0,1])/2;

%Imposing symmetry in z plane
a = 0:n1*(n2+1)-1;
a = ((n2+1)*(n1-1-floor(a/(n2+1)))+mod(a,n2+1)+1).';
p_array = (p_array + p_array(a,:)*[1,0,0;0,1,0;0,0,-1])/2;

quads = zeros(max(indices,[],"all"),4);
n_quad = 0;
for i = 1:n1-1
    for k = 1:n2
        n_quad = n_quad+1;
        quads(n_quad,1) = indices(k,i);
        quads(n_quad,2) = indices(k+1,i);
        quads(n_quad,3) = indices(k+1,i+1);
        quads(n_quad,4) = indices(k,i+1);
    end
end

imax = max(indices,[],"all");
clear a I i i_ j j_ k frc indices old_count p_new count_all x y z

%%
%Section to generate nodes in exposed patches
factor = 0; %Parameter determining blending of new and old directions, heuristic
p = zeros(n_points+1,3,n2+1);
p_len = zeros(1,n2+1);
check = @(p) (p(2)-p(3));
for i = 1:floor((n2+2)/2)
    q = zeros(n_points+1,3);
    i_ = (n1-1)*(n2+1)+i;
    q(1,:) = p_array(i_,:);
    dir = unit(p_array(i_,:)-p_array(i_-(n2+1),:));
    for j = 2:n_points+1
        curv_vect = principal_directions(q(j-1,:));
        [~,I] = max(abs(dir*curv_vect));
        dir = sign(dir*curv_vect(:,I))*curv_vect(:,I).';
        dir = unit(dir + factor*unit(p_array(i_,:)-p_array(i_-(n2+1),:)));
        q(j,:) = next_point_vector(q(j-1,:),lvl_init,ds,dir);
        if check(q(j,:))*check(q(j-1,:)) < 0
            a = -check(q(j-1,:))/check(q(j,:)-q(j-1,:));
            q(j,:) = plane_level(a*q(j,:)+(1-a)*q(j-1,:),lvl_init,ds,[0,1,-1]);
            break
        end
        dir = unit(q(j,:)-q(j-1,:));
    end
    %Selecting approximately 1 in frac nodes:
    p_len(1,[i,n2+2-i]) = (floor((j/frac)+0.5)+1)*[1,1];
    for k = 1:p_len(i)
        a = (j-1)*((k-1)/(p_len(i)-1))+1;
        p(k,:,i) = q(floor(a-0.5)+1,:); 
    end
end
p = p(1:max(p_len),:,:);
for i = n2/2:-1:2
    p(:,:,(n2+2)-i) = p(:,:,i)*[0,1,0;1,0,0;0,0,1];
end
p(:,:,n2+1) = p(:,:,1)*[0,1,0;1,0,0;0,0,1];
clear curv_vect q

%%
%Indexing points in p and adding them to p_list
p_count = max(cumsum(p_len));
p_list = zeros(p_count(end),3);
p_index = zeros(n2+1,max(p_len));
p_list(1:n2+1,:) = squeeze(p(1,:,:)).';
p_index(:,1) = 1:n2+1;
start_index = n2+1;
for i = 1:n2+1
    p_list(end-(n2+1)+i,:) = squeeze(p(p_len(i),:,i));
    p_index(i,p_len(i)) = p_count-(n2+1)+i;
    if p_len(i) > 2
        p_index(i,2:p_len(i)-1) = start_index + (1:p_len(i)-2);
        start_index = start_index + p_len(i)-2;
        for j = 2:p_len(i)-1
            p_list(p_index(i,j),:) = squeeze(p(j,:,i));
        end
    end
end

%%
%Creating elements in exposed patches
p_quad = zeros(p_count,4);
n_p_quad = 0;
tris = zeros(p_count,3);
n_tri = 0;
for i = 1:n2/2
    I = p_len(i);
    if p_len(i) ~= p_len(i+1)
        tris(n_tri+1,:) = [p_index(i,I),p_index(i+1,I),p_index(i+1,I+1)];
        tris(n_tri+2,:) = [p_index(n2+2-i,I),...
            p_index(n2+1-i,I),p_index(n2+1-i,I+1)];
        n_tri = n_tri + 2;
    end
    for j = 1:p_len(i)-1
        k = OR(0,1,j>=I);
        p_quad(n_p_quad+2*j-1,:) = [p_index(i,j),p_index(i,j+1),...
            p_index(i+1,j+k+1),p_index(i+1,j+k)];
        p_quad(n_p_quad+2*j,:) = [p_index(n2+2-i,j),p_index(n2+2-i,j+1),...
            p_index(n2+1-i,j+k+1),p_index(n2+1-i,j+k)];
    end
    n_p_quad = n_p_quad+2*(p_len(i)-1);
end

%Merging p_list and p_array
p_array = [p_array;p_list(n2+2:end,:)];
quads = [quads(1:n_quad,:);p_quad(1:n_p_quad,:)+imax-(n2+1)];
p_array = [p_list(end:-1:(n2+2),:)*[0,1,0;1,0,0;0,0,-1];p_array];
quads = [p_count+1-p_quad(1:n_p_quad,:);quads+p_count-(n2+1)];
tris = [p_count+1-tris(1:n_tri,:);tris(1:n_tri,:)+p_count+imax-2*(n2+1)];
imax = height(p_array);
n_quad = n_quad+2*n_p_quad;
n_tri = 2*n_tri;
clear a check factor I i_ i j k m n_p_quad p p_count p_index p_len p_list p_quad start_index

%%
%Section for rotating patches
for i = 1:11
    quads(n_quad*i+1:n_quad*(i+1),:) = quads(1:n_quad,:) + imax*i;
    tris(n_tri*i+1:n_tri*(i+1),:) = tris(1:n_tri,:) + imax*i;
    dir = [0,(1-2*floor(i/6))*floor(mod([i+3,i],6)/3)];
    A = rotmat([1,0,0],(pi/2)*floor(i/3))*rotmat(dir,(pi/2)*mod(i,3));
    p_array(imax*i+1:imax*(i+1),:) = p_array(1:imax,:)*A;
end
n_quad = n_quad*12; n_tri = n_tri*12;
clear A dir

%%
%Section for stitching adjacent patches
copies = zeros(12*imax,1);
indices = imax*floor((0:24*n2+23)/(2*n2+2))...
    + repmat([1:n2+1,imax-(n2:-1:0)],1,12);
for i = 1:width(indices)-2*(n2+1)
    if copies(indices(i)) == 0
        for j = i+1:width(indices)
            if max(abs(p_array(indices(j),:)-p_array(indices(i),:))) <= 0.1*Ds
                copies(indices(j)) = indices(i);
            end
        end
    end
end

I = 12*(imax-(n2+1))-4;
A = zeros(1+12*imax,1);
p_temp = zeros(I,3);
j = 0;
for i = 1:12*imax
    j = j + OR(1,0,copies(i)); %Counts unique elements
    p_temp(j,:) = OR(p_array(i,:),p_temp(j,:),copies(i));
    A(1+i) = OR(j,A(1+copies(i)),copies(i)); %Indices offset by 1 to avoid A(0)
end
A = A(2:end,1);

p_array = p_temp;
quads(1:n_quad,:) = remap(quads(1:n_quad,:), A);
tris(1:n_tri,:) = remap(tris(1:n_tri,:), A);
clear A copies imax i j k n1 n2 indices p p_temp

%%
%Triangle stitching
p = [quads(1:n_quad,:);zeros(floor(n_tri/2),4)]; 
k = floor(n_tri/12);
m = n_quad+1;
tris(1:n_tri,:) = sort(tris(1:n_tri,:),2,"ascend");
for i = 1:11*k   
    for j = i-mod(i-1,k)+k:n_tri
        l = check_tris(tris(i,:),tris(j,:));
        if l ~= 0
            p(m,:) = [tris(i,l(1)),tris(i,mod(l(1),3)+1),...
                tris(j,l(2)),tris(i,mod(l(1)+1,3)+1)];
            m = m+1;
            break;
        end
    end
end
n_quad = n_quad+floor(n_tri/2);
quads = p;
clear p i j k l m n_tri tris

%%
%Section for Laplacian smoothing
n_iter_smooth = 15; %Number of rounds of Laplacian smoothing, heuristic
Q = false(I); %Connectivity matrix
for i = 1:n_quad
    for j = 1:4
        Q(quads(i,j),quads(i,mod(j,4)+1)) = true;
        Q(quads(i,mod(j,4)+1),quads(i,j)) = true;
    end
end
n_deg = sum(Q,2);

a = 1:I;
for i = 1:n_iter_smooth
    p = p_array;
    for j = 1:I
        if abs(max(abs(p_array(j,:)))-0.5) < 0.1*Ds
            continue
        end
        p(j,:) = p_array(j,:) + sum((p_array(a(Q(j,:)),:)-p_array(j,:)),1)/n_deg(j,1);
        p(j,:) = grad_move(p(j,:),lvl_init);
    end
    p_array = p;
end
clear a i j n_deg n_iter_smooth p Q rounds

%%
%Section for traversing isolevels
if n_layers == 0
    if exp_ratio ~= 1
        n_layers = ceil(log(1+(BL_ht/l1_ht)*(exp_ratio-1))/log(exp_ratio));
    else
        n_layers = ceil((BL_ht/l1_ht));
    end
end
quads = [quads; zeros(n_layers*n_quad,4)];
p_array = [p_array; zeros(n_layers*I,3)];

for i = 1:n_layers
    for j = i*I+(1:I)
        lvl = f(p_array(j-I,:))...
            + ((-1)^mesh_dir)*(exp_ratio^(i-1))*(l1_ht/Size)*norm(grad(p_array(j-I,:)));
        p_array(j,:) = grad_move(p_array(j-I,:),lvl);
    end
    quads(i*n_quad+(1:n_quad),:) = quads(1:n_quad,:) + i*I;
end

%%
%Creating pyramid elements on the final layer
p_next = zeros(n_quad,3);
n_pyr = n_quad;
pyrs = zeros(n_pyr,5);
for i = (1:n_quad)
    p_ = p_array(quads(i+n_layers*n_quad,:)',:);
    p_next(i,:) = mean(p_,1);
    %Optimising pyramid element quality
    dir = unit(cross(p_(1,:)-p_(3,:),p_(2,:)-p_(4,:)));
    dir = -dir*sign(dot(p_next(i,:),dir))*((-1)^mesh_dir);
    z = norm(p_ - p_next(i,:),"fro")^2;
    z = z + norm(p_ - p_([2,3,4,1],:),"fro")^2;
    p_next(i,:) = p_next(i,:) + dir*sqrt(z/8);
end
pyrs(:,1:4) = quads((1:n_quad)+n_layers*n_quad,:); 
pyrs(:,5) = height(p_array) + (1:n_pyr)'; 
p_array = [p_array;p_next];

%Scaling unit cell dimension to the specified value
p_array = p_array*Size;
clear dir i j lvl p_ p_next z

%%
%Saving and visualisation
if to_mat
   save("to_gmsh.mat") %#ok<UNRCH>
end

if draw_graph
    figure(1) %#ok<UNRCH>
    view(ones(1,3))
    hold on;
    xlim(Size*[-0.5 0.5])
    ylim(Size*[-0.5 0.5])
    zlim(Size*[-0.5 0.5])
    for i =  1:n_quad*(n_layers+1)
        c = floor((i-1)/n_quad)/n_layers;
        p_ = p_array(quads(i,:).',:);
        fill3(p_(:,1),p_(:,2),p_(:,3),[c*(2/3), 0, 1-c*(2/3)])
    end
    clear c
    for i = 1:n_pyr
        for j = 1:4
            p_ = p_array(pyrs(i,[j,mod(j,4)+1,5]).',:);
            fill3(p_(:,1),p_(:,2),p_(:,3),"g")
        end
    end
end

%%
function value = f(p)
    value = cos(2*pi*p(1))+cos(2*pi*p(2))+cos(2*pi*p(3));
end

function value = fx(p) 
    value = -2*pi*sin(2*pi*p(1));
end

function value = fy(p) 
    value = -2*pi*sin(2*pi*p(2));
end

function value = fz(p) 
    value = -2*pi*sin(2*pi*p(3));
end

function value = fxx(p) 
    value = -4*pi^2*cos(2*pi*p(1));
end

function value = fyy(p) 
    value = -4*pi^2*cos(2*pi*p(2));
end

function value = fzz(p) 
    value = -4*pi^2*cos(2*pi*p(3));
end

function value = fxy(~) 
    value = 0;
end

function value = fyz(~) 
    value = 0;
end

function value = fzx(~) 
    value = 0;
end

function value = grad(p)
    value = [fx(p),fy(p),fz(p)];
end

function y = unit(x)
    y = x/norm(x);
end

%Corrects a point to the specified isolevel along a specified vector
function value = find_level(P,const,ds,dir)
    epsilon = 1e-12;
    fval = f(P) - const;
    while abs(fval) > epsilon
        while dot(dir,grad(P)) == 0
            P = P+dir*ds;
            fval = f(P) - const;
        end
        P = P - ds*fval*dir/dot(dir,grad(P));
        fval = f(P) - const;
    end
    value = P; 
end

%Corrects a point to the specified isolevel along a specified plane
function value = plane_level(P,const,ds,dir)
    epsilon = 1e-12;
    fval = f(P) - const;
    dir = unit(dir);
    while abs(fval) > epsilon
        g = grad(P);
        g = g - dot(g,dir)*dir;
        P = P - ds*fval*g/(g*g.');
        fval = f(P) - const;
    end
    value = P; 
end

function value = Hessian(p)
    x_ = fx(p);
    y_ = fy(p);
    z_ = fz(p);
    xx = fxx(p);
    yy = fyy(p);
    zz = fzz(p);
    xy = fxy(p);
    yz = fyz(p);
    zx = fzx(p);
    denominator = (x_^2+y_^2+z_^2)^(3/2);
    H11 = xx*y_^2 - x_*y_*xy + xx*z_^2 - x_*z_*zx; 
    H12 = xy*y_^2 - x_*y_*yy + xy*z_^2 - x_*z_*yz; 
    H13 = zx*y_^2 - x_*y_*yz + zx*z_^2 - x_*z_*zz;
    H21 = xy*x_^2 - x_*y_*xx + xy*z_^2 - y_*z_*zx;
    H22 = yy*x_^2 - x_*y_*xy + yy*z_^2 - y_*z_*yz;
    H23 = yz*x_^2 - x_*y_*zx + yz*z_^2 - y_*z_*zz;
    H31 = zx*x_^2 - x_*z_*xx + zx*y_^2 - y_*z_*xy;
    H32 = yz*x_^2 - x_*z_*xy + yz*y_^2 - y_*z_*yz; 
    H33 = zz*x_^2 - x_*z_*zx + zz*y_^2 - y_*z_*yz;
    value = [H11,H12,H13;H21,H22,H23;H31,H32,H33]/denominator;
end

function value = principal_directions(p)
    H = Hessian(p);
    n = unit(grad(p));
    [eig_vect,~] = eig(H);
    [~,I] = max(abs(n*eig_vect));
    value = eig_vect(:,[mod(I+1,3)+1,mod(I,3)+1]);
    for i = 1:2
        value(:,i) = real(unit(value(:,i) - dot(value(:,i),n)*n.'));
    end
end

%Find the next point on a given isolevel along a specified vector
function value = next_point_vector(p,const,ds,dir)
    epsilon = 1e-12;
    v = unit(dir);
    n = unit(grad(p));
    ds_cap = unit(v - (n*v.')*n);
    p = p + ds*ds_cap;
    fdash = f(p)-const;
    while (abs(fdash) > epsilon)
        n = grad(p);
        p = p - fdash*n/(n*n.');
        fdash = f(p)-const;
    end
    value = p;
end

function value = grad_move(p,level)
    epsilon = 1e-9;
    fval = f(p) - level;
    while abs(fval) > epsilon
        g = grad(p);
        p = p-0.1*fval*g/(g*g.');
        fval = f(p) - level;
    end
    value = p;
end

function c = OR(a,b,dir)
    if dir == 0
        c = a;
    else
        c = b;
    end
end

function A = rotmat(dir,theta)
    dir = unit(dir);
    A = [dir(1);dir(2);dir(3)]*[dir(1),dir(2),dir(3)];
    B = eye(3) - A;
    C = B*[0,dir(3),-dir(2);-dir(3),0,dir(1);dir(2),-dir(1),0];
    A = A + B*cos(theta) + C*sin(theta);
end

function list = remap(list,map)
    for i = 1:height(list)
        for j = 1:width(list)
            list(i,j) = map(list(i,j),1);
        end
    end
end

function odd_index = check_tris(t1,t2)
    odd_index = 0;
    combinations = [[2,3];[1,3];[1,2]];
    for i = 1:3
        for j = 1:3
            if max(abs(t1(combinations(i,:))-t2(combinations(j,:)))) == 0 
                odd_index = [i,j];
                return
            end
        end
    end
end

%Next point on a given isolevel traversing along a specified plane
function value = next_point_plane(p,const,ds,normal,dir)
    epsilon = 1e-12;
    n = unit([normal(1),normal(2),normal(3)]);
    g = unit(grad(p));
    ds_cap = unit(cross(n,g));
    p = p + ((-1)^dir)*ds*ds_cap;
    fdash = f(p)-const;
    while (abs(fdash) > epsilon)
        g = grad(p);
        p = p - fdash*unit(g - (g*n.')*n);
        fdash = f(p)-const;
    end
    value = p;
end

function q = pyr_qual(p) %p = [p1;p2;p3;p4;p5]
    p = p(1:4,:)-p(5,:);
    vol = abs(det(p(1:3,:))+det(p([1,3,4],:)))/6;
    l_sum = 0;
    for i = 1:4
        l_sum = l_sum + norm(p(i,:))^2 + norm(p(i,:)-p(mod(i,4)+1,:))^2;
    end
    q = 96*vol/(l_sum^(3/2));
end

function out = skew(p) %p = [p1;p2;p3;p4]
    theta = zeros(1,4);
    for i = 1:4
        theta(i) = acos(unit(p(mod(i,4)+1,:)-p(i,:))...
            *unit(p(mod(i-2,4)+1,:)-p(i,:)).');
    end
    out = (2/pi)*max(max(theta)-(pi/2),(pi/2)-min(theta));
end

function out = quality(p) %p = [p1;p2;p3;p4]
    area = (1/4)*(abs(det(p([4,1,2],:)))+abs(det(p([2,3,4],:)))+...
        abs(det(p([1,2,3],:)))+abs(det(p([1,3,4],:))));
    p = p-p([2,3,4,1],:);
    out = 4*area/(trace(p*p'));
end

function bool = same_face(p1,p2)
    [~,M1] = max(abs(p1));
    [~,M2] = max(abs(p2));
    bool = (M1*sign(p1(M1)) == M2*sign(p2(M2)));
end