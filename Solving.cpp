#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <random>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
using namespace std;

const double a = -1;       
const double b = 1;  
const double EPS = 1e-8; 
const int MAX_IT = 500;

const int K = 3;         
const int N = 3;   // узлов     
const int L = 10;       
const int M = (N-1)*K+1;
const int M_vis = 1000;


double func(double x);
void print(double value, int size, int space, ofstream &fout);
void mass(double *randd, double *x, double *y);
void deltas (double *abs, double *approx, int size, ofstream &fout, const string name);
void denominator(double znam[N], double *x);
double base(double tochka, double *x, double znam[N], int i, int seg);
void block(double *randd, double *x, double znam[N], double mat[][N], int seg);
void Gram(double **gram, double *randd, double *x, double znam[N]);
void Vector_b (double *B, double *randd, double *x, double znam[N]);
void SLAU(double **gram, double *B, double *c);

void tape(double **gram, double **tape, int row, int col);
double* gauss(double** mat1, double* B1, int row, int col);
double* LU (double **mat1, double *B1, int row, int col);
double* Choletsky(double **mat1, double *B1, int row);
double* Relax(double **mat, double *B, int row, double* r_rel);
double* Gradient(double **mat, double *B, int row, double *g);
double* multiply_sim(double *x, double **mat, int row);
double norm(double *vect, int size);

double func(double x){
    return pow(x, 3);
}

void print(double value, int size, int space, ofstream &fout){
    fout << scientific << setprecision(size) << setw(space) << right << value;
}
// Работаем с нижней частью матрицы (левой полулентой)
double* multiply_sim(double *x, double **mat, int row){
    int i, j;
    double* res = new double[row]();
    for (i = 0; i < row; i ++){
        res[i] += mat[i][N-1] * x[i];                       // диагональ, т.к в ленте оставили верхние диагонали
        for (j = 1; j < N && i+j < row; j ++){
            res[i] += mat[i+j][N-1-j] * x[i+j];                // верхние диагонали     
            res[i+j] += mat[i+j][N-1-j] * x[i];
        }
    }
    return res;
}

 
double norm(double *vect, int size){
    double l2 = 0.0;
    for (int i = 0 ; i < size; i ++)
        l2 += vect[i]*vect[i];
    return sqrt(l2);
}

void tape(double **gram, double **tape, int row, int col){
    int i, j, n;
    for (i = 0; i < row; i++){ 
        for (j = 0; j < col; j++){
            n = i+j;
            if (1-N+n > M-1 || 1-N+n < 0)
                tape[i][j] = 0;
            else
                tape[i][j] = gram[i][1-N+n];
        }
    }
}

double* gauss(double** mat1, double* B1, int row, int col){
    int i, j, k, ind;
    double tmp;
    // Копирование
    double **mat = new double* [row];
    for (i = 0; i < row; i ++){
        mat[i] = new double [col];
        memcpy(mat[i], mat1[i], col * sizeof(double));
    }
    double *B = new double [row];
    memcpy(B, B1, M * sizeof(double));
    // Приводим к верхнетреугольному виду
    for (k = 0; k < M; k ++){                                   // идем по столбцам исходной (по диагональным элементам)      
        for (i = k; i < min(row, k + N); i ++){                              // идем по строкам исходной   
            ind = N-1+k-i;
            // Деление 
            tmp = mat[i][ind];
            if (abs(tmp) < EPS){                                // на нуль делить нельзя
                continue;
            }
            for (j = ind; j < col; j ++){                        // идем по стобцам ленточной
                mat[i][j] /= tmp;
            }
            B[i] /= tmp;
            // Вычитание
            if (i == k){                                        // само из себя не вычитается 
                continue;
            }               
            for (j = k; j < min(row, k + N); j ++){            // идем по столбцам исходной: от диагонального до края ленты
                mat[i][N-1+j-i] -= mat[k][N-1+j-k];    
            }
            B[i] -= B[k];
        }
    }
    // Решаем систему
    double *x = new double[row]();
    for (k = row-1; k >= 0; k --){                            // идем по строкам исходной
        x[k] = B[k];
        i = k-1;
        for(i = max(0, k-N+1); i < k; i ++){                 // идем по строкам исходной с учетом ленты
            B[i] -= x[k] * mat[i][N-1+k-i];
        }
    }
    for (i = 0; i < row; i ++)
        delete [] mat[i];
    delete [] mat;
    delete [] B;
    return x; 
}

double* LU (double **mat1, double *B1, int row, int col){
    int i, j, k;
    double sum;
    // Копирование
    double **mat = new double* [row];
    for (i = 0; i < row; i ++){
        mat[i] = new double [col];
        memcpy(mat[i], mat1[i], col*sizeof(double));
    }
    double *B = new double [M];
    memcpy(B, B1, M*sizeof(double));
    // Записываем L, U в исходную матрицу
    for (k = 0; k < M; k ++){
        for (i = k+1; i <= min(M-1, k+N-1); i ++){
            if (abs(mat[k][N-1]) < EPS){
                cout << "Error\n";
                return nullptr;
            }
            mat[i][N-1+k-i] /= mat[k][N-1];
            for (j = max(k+1, i-(N-1)); j <= min(M-1, i+N-1); j ++){
                if (j <= k+N-1){
                    mat[i][N-1+j-i] -= mat[i][N-1+k-i]*mat[k][N-1+j-k];
                }
            }
        }  
    }
    // Ly = b
    double *y = new double [M];
    for (i = 0; i < M; i ++) {
        sum = B[i];
        for (j = max(0, i-(N-1)); j < i; j ++) {
            sum -= mat[i][N-1+j-i] * y[j];  
        }
        y[i] = sum;  // L[i,i] = 1
    }
    // Ux = y
    double *x = new double [M];
    for (i = M-1; i >= 0; i --) {
        sum = y[i];
        for (j = i+1; j <= min(M-1, i+N-1); j ++) {
            sum -= mat[i][N-1+j-i]*x[j];  
        }
        if (abs(mat[i][N-1]) < EPS){
            cout << "Error\n";
            return nullptr;
        }
        x[i] = sum/mat[i][N-1];  // Делим на диагональ U
    }
    for (i = 0; i < row; i ++)
        delete [] mat[i];
    delete [] mat;
    delete [] B;
    return x;
}  
// Работаем с нижней частью матрицы (левой полулентой)
double* Choletsky(double **mat1, double *B1, int row){
    int i, j, k, ind;
    double sum;
    // Копирование
    double **mat = new double* [row];
    for (i = 0; i < row; i ++){
        mat[i] = new double [N];
        memcpy(mat[i], mat1[i], N * sizeof(double));
    }
    double *B = new double [M];
    memcpy(B, B1, M * sizeof(double));

    for (int i = 0; i < row; i ++){
        // Диагональ
        sum = mat[i][N-1];
        for (k = max(0, i-(N-1)); k < i; k ++){
            ind = k-i+N-1;
            sum -= mat[i][ind] * mat[i][ind];
        }
        mat[i][N-1] = sqrt(sum);
        // Внедиагональ
        for (j = i+1; j <= min(row-1, i+N-1); j ++){
            sum = mat[j][N-1+i-j];
            for (k = max(0, j-(N-1)); k < i; k ++) {
                sum -= mat[j][k-j+N-1] * mat[i][k-i+N-1];
            }
            if (abs(mat[i][N-1]) < EPS){
                cout << "Error\n";
                return nullptr;
            }
            mat[j][i-j+N-1] = sum / mat[i][N-1];
        }
    }
    // L*y = b
    double *y = new double [row];
    for (i = 0; i < row; i ++){
        y[i] = B[i];
        for (j = max(0, i-(N-1)); j < i; j ++){
            y[i] -= mat[i][j-i+N-1] * y[j];
        }
        if (abs(mat[i][N-1]) < EPS){
            cout << "Error\n";
            return nullptr;
        }
        y[i] /= mat[i][N-1];
    }
    // L^T*x = y
    double *x = new double [row];
    for (i = row-1; i >= 0; i --){
        x[i] = y[i];
        for (j = i+1; j <= min(row-1, i+N-1); j ++){
            x[i] -= mat[j][i-j+N-1] * x[j];
        }
        if (abs(mat[i][N-1]) < EPS){
            cout << "Error\n";
            return nullptr;
        }
        x[i] /= mat[i][N-1];
    }
    return x;
}
// Работаем с нижней частью матрицы (левой полулентой)
double* Relax(double **mat, double *B, int row, double* r_rel){
    int i, j, it = 0, flag = 1;
    double* x = new double [row](), *x_old = new double[row]();
    double omega = 1.2, delta = 1e-16; 
    double sum1, sum2;

    for (it = 0; it < MAX_IT; it ++){
        memcpy(x_old, x, row * sizeof(double));
        for (i = 0; i < row; i ++){
            sum1 = 0.0; sum2 = 0.0;
            for (j = max(0, i-(N-1)); j < i; j ++){
                sum1 += mat[i][j-i+N-1] * x[j];
            }
            for (j = i+1; j <= min(row-1, i+N-1); j ++){
                sum2 += mat[j][i-j+N-1] * x_old[j];
            }
            if (abs(mat[i][N-1]) < EPS){
                cout << "Error\n";
                return nullptr;
            }
            x[i] = (1.0-omega)*x_old[i] + omega*(B[i] - sum1 - sum2)/mat[i][N-1];    
            r_rel[i] = fabs(x[i] - x_old[i]);
        }
        if (norm(r_rel, row) <= EPS*norm(x, row) + delta){
            break;
        }
    }
    cout << "iterations of relaxation: " << it + 1 << endl;
    delete [] x_old;
    return x;
}

double* Gradient(double **mat, double *B, int row, double *g){
    int i, it = 0;
    double alpha, beta, pg, py, gy;
    double *x = new double [M](), *y = new double [M](), *p = new double[M](); 
    for (i = 0; i < M; i ++){
        g[i] = -B[i];
        p[i] = -g[i];
    }
    for (it = 0; it < MAX_IT; it ++){
        y = multiply_sim(p, mat, row);
        for (i = 0; i < M; i ++){
            pg += p[i]*g[i];
            py += p[i]*y[i];
        }
        if (abs(py) < EPS){
            cout << "Error\n";
            return nullptr;
        }
        alpha = -pg/py;
        for(i = 0; i < M; i ++){
            x[i] += alpha*p[i];
            g[i] += alpha*y[i];
        }
        for(i = 0; i < M; i ++)  
            gy += g[i]*y[i];
        beta = gy/py;                       // py != 0 -- проверили выше
        for(i = 0; i < M; i ++) 
            p[i] = -g[i] + beta*p[i];  
        if (norm(g, row) <= EPS*norm(B, row)){
            break;
        }        
    }
    cout << "iterations of gradient: " << it + 1<< endl;
    delete [] y; delete [] p;
    return x;
}

void deltas (double *abs, double *diff, int size, ofstream &fout, const string name){
    int i;
    double tmp;
    double otn_l1 = 0.0, otn_l2 = 0.0, otn_oo = fabs(abs[0]);
    double abs_l1 = 0.0, abs_l2 = 0.0, abs_oo = fabs(diff[0]);
    for (i = 0; i < size; i ++){
        tmp = fabs(diff[i]);
        abs_l1 += tmp;
        abs_l2 += tmp * tmp;
        if (tmp > abs_oo)
            abs_oo = tmp;
        tmp = fabs(abs[i]);
        otn_l1 += tmp;              // считаем знамнатель относительной погрешности
        otn_l2 += tmp * tmp;
        if (tmp > otn_oo)
            otn_oo = tmp;
    } 
    abs_l2 = sqrt(abs_l2);
    otn_l2 = abs_l2 / sqrt(otn_l2);
    otn_l1 = abs_l1 / otn_l1;
    otn_oo = abs_oo / otn_oo;
    
    fout << "\n----- " << name << " -----\n";
    fout << "otn_l1 = "; print(otn_l1, 3, 9, fout); fout << "      abs_l1 = "; print(abs_l1, 3, 9, fout); fout << "\n"; 
    fout << "otn_l2 = "; print(otn_l2, 3, 9, fout); fout << "      abs_l2 = "; print(abs_l2, 3, 9, fout); fout << "\n"; 
    fout << "otn_oo = "; print(otn_oo, 3, 9, fout); fout << "      abs_oo = "; print(abs_oo, 3, 9, fout); fout << "\n"; 
}

void mass(double *randd, double *x, double *y){    
    int i, j = 0, k, begin;
    double res, delta = 1e-16;
    random_device rand;   
    mt19937 gen(rand());  
    for (k = 0; k < K; k ++){
        begin = k*(N-1);
        uniform_real_distribution<> seg(x[begin],x[begin+N-1]); 
        for (i = 0; i < L; i ++){
            res = seg(gen);
            if (res-x[begin] > delta && x[begin+N-1]-res > delta){
                randd[j] = res;
                y[j] = func(randd[j]);
                j ++;
            }
        }
    }
}

void denominator(double znam[N], double *x){
    int i, j;
    for (i = 0; i < N; i ++){
        znam[i] = 1.0;
        for (j = 0; j < N; j ++)
            if (i != j)
                znam[i] *= (x[i]-x[j]);
    }
}

double base(double tochka, double *x, double znam[N], int i, int seg){
    double res = 1.0;
    int j, begin = seg*(N-1);
    for (j = 0; j < N; j ++){
        if (j != i){
            res *= (tochka-x[begin+j]);
        }
    }
    return res/znam[i];
}

void block(double *randd, double *x, double znam[N], double mat[][N], int seg){
    int i, j, m, ind;
    double phi1, phi2;
    for (i = 0; i < N; i ++){
        for (j = 0; j < N; j ++){
            mat[i][j] = 0.0;
            for (m = 0; m < L; m ++){
                ind = seg*L + m; 
                phi1 = base(randd[ind], x, znam, i, seg);
                phi2 = base(randd[ind], x, znam, j, seg);
                mat[i][j] += phi1 * phi2;
            }
        }
    }        
}

void Gram(double **gram, double *randd, double *x, double znam[N]){
    int i, j, k;
    double mat[N][N];
    for (i = 0; i < M; i ++)
        for (j = 0; j < M; j ++)
            gram[i][j] = 0.0;

    for (k = 0; k < K; k ++){
        block(randd, x, znam, mat, k);
        for (i = 0; i < N; i ++){
            for (j = 0; j < N; j++){
                gram[i+(N-1)*k][j+(N-1)*k] += mat[i][j];
            }
        }
    }
}

void Vector_b (double *B, double *randd, double *x, double znam[N]){
    int i, m, k, ind;
    double phi, f;
    for (i = 0; i < M; i ++)
        B[i] = 0.0;
    for (k = 0; k < K; k ++){
        for (i = 0; i < N; i ++){
            ind = i+(N-1)*k;
            for (m = 0; m < L; m ++){
                phi = base(randd[k*L+m], x, znam, i, k);
                f = func(randd[k*L+m]);
                B[ind] += phi*f;
            }
        }
    }
}

void SLAU(double **gram, double *B, double *c){
    int i, j;
    Eigen::MatrixXd A(M, M);
    for (i = 0; i < M; i ++)
        for (j = 0; j < M; j ++)
            A(i, j) = gram[i][j];
    Eigen::VectorXd b(M);
    for (i = 0; i < M; i ++)
        b(i) = B[i];
    // решение слау -- находим коэффициенты для наилучшего приближения
    Eigen::VectorXd x = A.fullPivLu().solve(b);
    for (i = 0; i < M; i ++)
        c[i] = x(i);
}

int main(){
    int i, j, size = L*K;
    double h = (b-a)/(M-1);
    double *x = new double [M], *y = new double [size], *randd = new double [size];
    
    ofstream fout("gramm.txt");
    for (i = 0; i < M; i ++)
        x[i] = a + i*h;
    mass(randd, x, y);
    double znam[N] = {0.0};
    denominator(znam, x);
    double *B = new double [M];
    double *c = new double [M];
    double **gram = new double* [M];
    for (i = 0; i < M; i ++)
        gram[i] = new double [M];
    Gram(gram, randd, x, znam);
    Vector_b(B, randd, x, znam);
    SLAU(gram, B, c);
   
    int col = 2*N-1, row = M;
    double **t = new double* [row], **t_sim = new double* [row];
    for (i = 0; i < row; i ++){
        t[i] = new double [col];
        t_sim[i] = new double [N];
    }
    tape(gram, t, row, col);
    for (i = 0; i < row; i ++)
        memcpy(t_sim[i], t[i], N * sizeof(double));
    double *x_gauss = new double [M], *x_lu = new double [M], *x_ch = new double [M];
    double *x_gr = new double [M], *g = new double [M], *x_rel = new double [M], *r_rel = new double [M];
    x_gauss = gauss(t, B, row, col);
    x_lu = LU(t, B, row, col);
    x_ch = Choletsky(t, B, row);
    x_rel = Relax(t, B, row, r_rel);
    x_gr = Gradient(t_sim, B, row, g);
    
    ////////////////// Погрешности /////////////
    ofstream fout2("deltas.txt");
    double *diff = new double [M];

    for (i = 0; i < M; i ++) diff[i] = x_gauss[i] - c[i];
    deltas(c, diff, M, fout2, "Gauss");
    for (i = 0; i < M; i ++) diff[i] = x_lu[i] - c[i];
    deltas(c, diff, M, fout2, "LU");
    for (i = 0; i < M; i ++) diff[i] = x_ch[i] - c[i];
    deltas(c, diff, M, fout2, "Choletsky");
    for (i = 0; i < M; i ++) diff[i] = x_rel[i] - c[i];
    deltas(c, diff, M, fout2, "Relaxation");
    for (i = 0; i < M; i ++) diff[i] = x_gr[i] - c[i];
    deltas(c, diff, M, fout2, "Gradient");

    fout2 << endl << "---------------------- residual ----------------------" << endl;
    diff = multiply_sim(c, t_sim, M);
    for (i = 0; i < M; i ++) diff[i] -= B[i];
    deltas(B, diff, M, fout2, "Exact");

    diff = multiply_sim(x_gauss, t_sim, M);
    for (i = 0; i < M; i ++) diff[i] -= B[i];
    deltas(B, diff, M, fout2, "Gauss");

    diff = multiply_sim(x_lu, t_sim, M);
    for (i = 0; i < M; i ++) diff[i] -= B[i];
    deltas(B, diff, M, fout2, "LU");

    diff = multiply_sim(x_ch, t_sim, M);
    for (i = 0; i < M; i ++) diff[i] -= B[i];
    deltas(B, diff, M, fout2, "Choletsky");

    deltas(B, r_rel, M, fout2, "Relaxation");
    deltas(B, g, M, fout2, "Gradient");
   
    for (i = 0; i < M; i ++){
        for (j = 0; j < M; j ++)
            print(gram[i][j], 6, 15, fout);
        fout << "\n";
    }
    delete [] x; delete [] y; delete [] randd;
    for (i = 0; i < M; i++)
        delete[] gram[i]; 
    delete[] gram; 
    delete [] B; delete [] c;
    for (i = 0; i < row; i ++)
        delete [] t[i];
    delete [] t;
    delete [] x_gauss; delete [] x_lu; delete [] x_ch; delete [] x_rel; delete [] x_gr;
    delete [] g; delete [] r_rel; delete [] diff;
    
    fout.close(); fout2.close(); 
    return 0;
}