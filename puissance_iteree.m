clear variables;clc
% tolerance relative minimum pour l'ecart entre deux iteration successives 
% de la suite tendant vers la valeur propre dominante 
% (si |lambda-lambda_old|/|lambda_old|<eps, l'algo a converge)
eps = 1e-8;
% nombre d iterations max pour atteindre la convergence 
% (si i > kmax, l'algo finit)
imax = 5000; 

% Generation d une matrice rectangulaire aleatoire A de taille n x p.
% On cherche le vecteur propre et la valeur propre dominants de AA^T puis
% de A^TA
n = 1500; p = 500;
A = 5*randn(n,p);

% AAt, AtA sont deux matrices carrees de tailles respectives (n x n) et 
% (p x p). Elles sont appelees "equations normales" de la matrice A
AAt = A*A'; AtA = A'*A;

%% Methode de la puissance iteree pour la matrice AAt de taille nxn
% Point de depart de l'algorithme de la puissance iteree : un vecteur
% aleatoire, normalise
x = ones(n,1); x = x/norm(x);

cv = false;
iv1 = 0;        % pour compter le nombre d'iterations effectuees
t_v1 = cputime; % pour calculer le temps d execution de l'algo
err1 = 0;       % indication que le calcul est satisfaisant
                % on stoppe quand err1 < eps 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CODER L'ALGORITHME DE LA PUISSANCE ITEREE POUR LA MATRICE AAt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M1 = AAt;
lambda1 = x'*M1*x;
while(~cv)
   mu = lambda1;
   x = M1*x;
   x = x / norm(x);
   lambda1 = x'*M1*x;
   iv1 = iv1 + 1;
   err1 = norm(lambda1 - mu)/norm(mu);
   cv = (err1 <= eps) | (iv1 >= imax);
end
t_v1 = cputime-t_v1; % t_version1 : temps d execution de l algo de la 
                     % puissance iteree pour la matrice AAt

%% Methode de la puissance iteree pour la matrice AtA de taille pxp
% Point de depart de l'algorithme de la puissance iteree : un vecteur
% aleatoire, normalise
y = ones(p,1); 
y = y/norm(y);
M2 = AtA;
cv = false;
lambda2 = y'*M2*y; 
iv2 = 0;        % pour compter le nombre d iterations effectuees
t_v2 = cputime; % pour calculer le temps d execution de l'algo
err2 = 0;       % indication que le calcul est satisfaisant
                % on stoppe quand err2 < eps 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CODER L ALGORITHME DE LA PUISSANCE ITEREE POUR LA MATRICE AtA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while(~cv)
   mu = lambda2;
   y = M2*y;
   y = y/norm(y);
   lambda2 = y'*M2*y;
   iv2 = iv2 + 1;
   err2 = norm(lambda2 - mu)/norm(mu);
   cv = (err2 <= eps) | (iv2 >= imax);
end
t_v2 = cputime-t_v2;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APRES VOUS ETRE ASSURE DE LA CONVERGENCE DES DEUX METHODES, AFFICHER 
% L'ECART RELATIF ENTRE LES DEUX VALEURS PROPRES TROUVEES, ET LE TEMPS
% MOYEN PAR ITERATION POUR CHACUNE DES DEUX METHODES. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Erreur pour la methode avec la grande matrice = %0.3e\n', err1);
fprintf('Erreur pour la methode avec la petite matrice = %0.3e\n', err2);
fprintf('Ecart relatif entre les deux valeurs propres trouvees = %1.2e\n', 0)
fprintf('Temps pour une ite avec la grande matrice = %0.3e\n', t_v1/iv1);
fprintf('Temps pour une ite avec la petite matrice = %0.3e\n', t_v2/iv2);


t_v3 = cputime;
D = eig(AAt);
D = sort(D, 'descend');
t_v3 = cputime-t_v3; 
fprintf('Temps pour calculer la valeur propre maximale en utilisant la fonction eig et sort = %0.3e\n', t_v3);

fprintf('\nValeur propre dominante (methode avec la grande matrice) = %0.3e\n', lambda1);
fprintf('Valeur propre dominante (methode avec la petite matrice) = %0.3e\n', lambda2);
fprintf('Valeur propre dominante (fonction eig) = %0.3e\n', D(1));
