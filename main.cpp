/*
 *  Equação de Burger
 *
 *        U_t + U*U_x = v U_xx     , U=U(x,t),  a < x < b,     t>0
 *
 *        U_t = DU/dt   , U_x = DU/dx   e U_xx = D(DU/dx)/dx
 * */

#include <iostream>
#include <cmath>
#include <mpi.h>

using namespace std;

typedef double (*FPtr)(double, double);

char caracter;
double alfa = 5.0;
double beta = 4.0;
double v = 0.05;

double x0 = 0.0;
double x1 = 1.0;
double t0 = 0.0;
double t1 = 1.0;

int M = 100;
int N = 25;
const int itMax = 80;
double h;
double k;
double sigma;

int world_size; // número total de processos
int world_rank; // ID (rank) do processo

FPtr Uptr, EsqPtr, DirPtr;

bool imprime = false; //true;
bool imprime2 = false; //true;
bool imprime3 = false; //true;
double *w1 = nullptr;
double *wAux = nullptr;
double *F = nullptr;
double *S1 = nullptr;
double *S2 = nullptr;
double **w = nullptr;
double **DF = nullptr;
double **DF1 = nullptr;
double **DF2 = nullptr;

double DirFuncaoInicial(double x, double t);

double EsqFuncaoInicial(double x, double t);

double U(double x, double t);

void imprimeFuncao(char *nome, double x, double t, FPtr Uptr);

void NewMatriz(int nLin, int nCol, double **&M, double m);

void NewVetor(int nCol, double *&V, double v);

void imprimeMatriz(char *nome, int nlin, int ncol, double **&A);

void imprimeVetor(char *nome, int ncol, double *&v);

void inicializa(double **w, FPtr Uptr);

void copiaVetor(int ini, int fim, double *&vOrig, double *&vDest);

void copiaColMatrizToVetor(int col, int linIni, int linFim, double **&MOrig, double *&vDest);

void CalculaDF1(double **&DF1);

void CalculaDF2(double **&DF2, double *&w1);

void somaMatrizes(double **&S, double **&A, double **&B);

void escalar_x_matriz(double escalar, double **&A);

void Matriz_x_vetor(double *vR, double **A, double *v);

void CalculaF(int col, double *F, double **w, double *wAux);

void CondicaoDeDirichlet(int n, double **DF, double *F, double *w1, FPtr DirPtr, FPtr EsqPtr);

void copiaVetorToColMatriz(int col, double *v, double **A);

void ZeraMatriz(double **A);

void Jacobi(double **DF, double *F, double *S1, double *S2);

void w1_igual_w1_menos_S1(double *w1, double *S1);

void imprimeMatriz2(int lin, int col, double **A);

void imprimeMatriz3(int col, double **A);

void DeleteData(void);

double Burg(int M, int N);

int main() {

    MPI_Init(NULL, NULL); // Inicializa o MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


    if (world_rank == 0) {
        printf("    h          k        U(x=0.5;t=1)        w(0.5;1)         erro = U - w  ");
    }

    for (int l = 1; l < 4; l++) {
        if (l>1)
            N=N*2;
        /*
        for (int P = 0; P < world_size; P++) {
            if (world_rank == P) {
                Aprox = Burg(M, pow(2, world_rank) * N);
                printf("\n%\8.6f    %8.6f   %14.12f    %14.12f    %14.12f", h, k, R, Aprox, R - Aprox);
            }
        }
        */
        if (imprime2) {
            printf("\nworld_rank = %d", world_rank);
        }
        if (M > 1) {
            h = (x1 - x0) / M;
            k = (t1 - t0) / N;
        } else {
            h = M;
            k = N;
            M = (x1 - x0) / h;
            N = (t1 - t0) / k;
        }

        sigma = v * k / (h * h);
        Uptr = &U;
        EsqPtr = &EsqFuncaoInicial;
        DirPtr = &DirFuncaoInicial;
        if (imprime) {
            imprimeFuncao("Funcao U(x=0.5; t=1) =", 0.5, 1, Uptr);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        NewMatriz(M + 1, M + 1, DF, 0);
        NewMatriz(M + 1, M + 1, DF1, 0);
        NewMatriz(M + 1, M + 1, DF2, 0);
        NewMatriz(M + 1, N + 1, w, 0);
        NewVetor(M + 1, F, 0);
        NewVetor(M + 1, S1, 1);
        NewVetor(M + 1, S2, 1);
        NewVetor(M + 1, w1, 0);
        NewVetor(M + 1, wAux, 0);
        MPI_Barrier(MPI_COMM_WORLD);

        if (imprime) {
            imprimeMatriz("Matriz DF =", M + 1, M + 1, DF);
            imprimeMatriz("Matriz DF1 =", M + 1, M + 1, DF1);
            imprimeMatriz("Matriz DF2 =", M + 1, M + 1, DF2);
            imprimeMatriz("Matriz w =", M + 1, N + 1, w);
            imprimeVetor("Transposta do vetor F =", M + 1, F);
            imprimeVetor("Transposta do vetor w1 =", M + 1, w1);
        }

        inicializa(w, Uptr);
        copiaColMatrizToVetor(0, 0, M, w, w1);// w1 <-- w

        for (int n = 0; n < N; n++) { // N  passos no tempo

            for (int it = 1; it <= 6; ++it) { // iterações de Newton
                ZeraMatriz(DF1);
                ZeraMatriz(DF2);
                if (imprime) {
                    imprimeMatriz("Matriz w =", M + 1, N + 1, w);
                    imprimeVetor("Transposta do vetor w1 =", M + 1, w1);
                }

                CalculaDF1(DF1);
                CalculaDF2(DF2, w1);
                somaMatrizes(DF, DF1, DF2);

                if (imprime) {
                    imprimeMatriz("Matriz DF1 =", M + 1, M + 1, DF1);
                    imprimeMatriz("Matriz DF2 =", M + 1, M + 1, DF2);
                    imprimeMatriz("Matriz DF =", M + 1, M + 1, DF);
                }

                escalar_x_matriz(0.5, DF2);
                somaMatrizes(DF1, DF1, DF2);
                Matriz_x_vetor(wAux, DF1, w1); // wAux <-- DF1 * w1
                CalculaF(n, F, w, wAux);
                //imprimeMatriz("Matriz antes de Dirichelet DF =", M + 1, M + 1, DF);
                CondicaoDeDirichlet(n, DF, F, w1, DirPtr, EsqPtr);

                if (imprime) {
                    imprimeMatriz("Matriz DF2 =", M + 1, M + 1, DF2);
                    imprimeMatriz("Matriz DF1 =", M + 1, M + 1, DF1);
                    imprimeVetor("Transposta do vetor F =", M + 1, F);
                }
                //imprimeMatriz("Matriz Antes do Jacobi DF =", M + 1, M + 1, DF);
                Jacobi(DF, F, S1, S2);
                //imprimeVetor("Transposta do vetor S2 =", M + 1, S2);
                w1_igual_w1_menos_S1(w1, S2);
                //imprimeVetor("Transposta do vetor w1 =", M + 1, w1);

            } // Fim da iteração( Método de Newton)

            //w1_menos_S2(w1, S2);
            //w(:,n+1)=w1
            copiaVetorToColMatriz(n + 1, w1, w);


        }// Fim dos passos N
        if (world_rank == 0) {
            double R = U(0.5, 1);
            double Aprox = w[M / 2][N];
            printf("\n%\8.6f    %8.6f   %14.12f    %14.12f    %14.12f", h, k, R, Aprox, R - Aprox);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        DeleteData();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if ((imprime2) && (world_rank == 0)) {
        cout << "\n\nTecle uma tecla e apos Enter para finalizar...\n";
        cin >> caracter;
    }

    MPI_Finalize();
    return 0;
} // Fim main

double Burg(int M, int N) {

    if (imprime2 == 0) {
        printf("\nworld_rank = %d", world_rank);
    }
    if (M > 1) {
        h = (x1 - x0) / M;
        k = (t1 - t0) / N;
    } else {
        h = M;
        k = N;
        M = (x1 - x0) / h;
        N = (t1 - t0) / k;
    }

    sigma = v * k / (h * h);
    Uptr = &U;
    EsqPtr = &EsqFuncaoInicial;
    DirPtr = &DirFuncaoInicial;
    if (imprime) {
        imprimeFuncao("Funcao U(x=0.5; t=1) =", 0.5, 1, Uptr);
    }

    NewMatriz(M + 1, M + 1, DF, 0);
    NewMatriz(M + 1, M + 1, DF1, 0);
    NewMatriz(M + 1, M + 1, DF2, 0);
    NewMatriz(M + 1, N + 1, w, 0);
    NewVetor(M + 1, F, 0);
    NewVetor(M + 1, S1, 1);
    NewVetor(M + 1, S2, 1);
    NewVetor(M + 1, w1, 0);
    NewVetor(M + 1, wAux, 0);


    if (imprime) {
        imprimeMatriz("Matriz DF =", M + 1, M + 1, DF);
        imprimeMatriz("Matriz DF1 =", M + 1, M + 1, DF1);
        imprimeMatriz("Matriz DF2 =", M + 1, M + 1, DF2);
        imprimeMatriz("Matriz w =", M + 1, N + 1, w);
        imprimeVetor("Transposta do vetor F =", M + 1, F);
        imprimeVetor("Transposta do vetor w1 =", M + 1, w1);
    }

    inicializa(w, Uptr);
    copiaColMatrizToVetor(0, 0, M, w, w1);

    for (int n = 0; n < N; n++) { // N  passos no tempo

        for (int it = 1; it <= 2; ++it) { // iterações de Newton
            ZeraMatriz(DF1);
            ZeraMatriz(DF2);
            if (imprime) {
                imprimeMatriz("Matriz w =", M + 1, N + 1, w);
                imprimeVetor("Transposta do vetor w1 =", M + 1, w1);
            }

            CalculaDF1(DF1);
            CalculaDF2(DF2, w1);
            somaMatrizes(DF, DF1, DF2);

            if (imprime) {
                imprimeMatriz("Matriz DF1 =", M + 1, M + 1, DF1);
                imprimeMatriz("Matriz DF2 =", M + 1, M + 1, DF2);
                imprimeMatriz("Matriz DF =", M + 1, M + 1, DF);
            }

            escalar_x_matriz(0.5, DF2);
            somaMatrizes(DF1, DF1, DF2);
            Matriz_x_vetor(wAux, DF1, w1);
            CalculaF(n, F, w, wAux);
            //imprimeMatriz("Matriz antes de Dirichelet DF =", M + 1, M + 1, DF);
            CondicaoDeDirichlet(n, DF, F, w1, DirPtr, EsqPtr);

            if (imprime) {
                imprimeMatriz("Matriz DF2 =", M + 1, M + 1, DF2);
                imprimeMatriz("Matriz DF1 =", M + 1, M + 1, DF1);
                imprimeVetor("Transposta do vetor F =", M + 1, F);
            }
            //imprimeMatriz("Matriz Antes do Jacobi DF =", M + 1, M + 1, DF);
            Jacobi(DF, F, S1, S2);
            //imprimeVetor("Transposta do vetor S2 =", M + 1, S2);
            w1_igual_w1_menos_S1(w1, S2);
            //imprimeVetor("Transposta do vetor w1 =", M + 1, w1);

        } // Fim da iteração( Método de Newton)

        //w1_menos_S2(w1, S2);
        //w(:,n+1)=w1
        copiaVetorToColMatriz(n + 1, w1, w);


    }// Fim dos passos N

    double resultado = w[M / 2][N];
    DeleteData();

    return resultado;
}


void DeleteData() {

    delete[] w1;
    delete[] wAux;
    delete[] F;
    delete[] S1;
    delete[] S2;
    delete[] w;
    delete[] DF;
    delete[] DF1;
    delete[] DF2;
    //sleep(1);

}

void imprimeMatriz3(int col, double **A) {
    for (int i = 0; i <= M; i++)
        printf("\n%12.10f", A[i][col]);
}

void imprimeMatriz2(int lin, int col, double **A) {
    printf("%7.4f", A[lin][col]);
}


/*
 *    Subtração de vetores:  w1 <-- w1 - S1
 *    for (int i = 0; i <= M; i++)
 *        w1[i] = w1[i] - S1[i];
 */
void w1_igual_w1_menos_S1(double *w1, double *S1) {
    for (int i = 0; i <= M; i++)
        w1[i] = w1[i] - S1[i];
}

/* Método iterativo de Jacobi para resolver o sistema  DF*s = F
 *
 *
 * */
void Jacobi(double **DF, double *F, double *S1, double *S2) {
    // malha espacial local

    int tam = int(M / world_size);
    if (imprime3)
        printf(" tam = %d ", tam);
    int ip = world_rank * tam;
    int fp = (world_rank + 1) * tam;
    if (world_rank == world_size - 1)
        fp = M + 1;
    int my_I = fp - ip;

    double s;
    for (int n = 1; n <= itMax; n++) {

        for (size_t i = ip; i < fp; ++i) {
            s = 0.0;
            for (int j = 0; j <= M; j++) {
                if (j != i)
                    s = s + DF[i][j] * S1[j];
            }
            S2[i] = (F[i] - s) / DF[i][i];
            if (imprime2) {
                printf("\nAntes BCast n= %d, world_rank=%d, ip=%d, fp=%d, S2[%d] = %8.4f ,  My_I = %d", n, world_rank,
                       ip, fp, i, S2[i], my_I);
            }

        }
        MPI_Barrier(MPI_COMM_WORLD);
        /* everyone calls bcast, data is taken from root and ends up in everyone's buf */
        //MPI_Bcast(&S2[ip], my_I, MPI_DOUBLE, world_rank, MPI_COMM_WORLD);
        for (int i = 0; i < world_size - 1; i++) {
            MPI_Bcast(&S2[i * tam], tam, MPI_DOUBLE, i, MPI_COMM_WORLD);
        }
        // MPI_Bcast do último processo
        MPI_Bcast(&S2[(world_size - 1) * tam], M + 1 - (world_size - 1) * tam, MPI_DOUBLE, world_size - 1,
                  MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        if ((imprime2) && (world_rank == 0)) {
            for (int i = 0; i <= M; i++)
                printf("\nDepois BCast n= %d, world_rank=%d, ip=%d, fp=%d, S2[%d] = %8.4f ,  My_I = %d", n, world_rank,
                       ip, fp, i, S2[i], my_I);
        }

        for (int k = 0; k <= M; k++)
            S1[k] = S2[k];
    }
}

/*
 *  Inicializa a malha w[i][0] com a função inicial
 *  w[i][0] = Uptr(x0 + i * h, 0) para 0<=i<=M
 */
void inicializa(double **w, FPtr Uptr) {
    for (int i = 0; i <= M; i++)
        w[i][0] = Uptr(x0 + i * h, 0);
}

/*
 *  Entrada:
 *  a1 = n
 *  a2 = v
 *  a3 = x
 */

double DirFuncaoInicial(double x, double t) {
    return 0.0;
}

double EsqFuncaoInicial(double x, double t) {
    return 0.0;
}

/* Solução U(x,t) e função inicial U(x,0)
 * U(x,t)=[2 v beta pi sen(pi x) exp(-v t pi^2)]/[alfa + beta cos(pi x) exp(-v t pi^2)]
 */
double U(double x, double t) {

    return (2 * v * beta * M_PI * sin(M_PI * x) * exp(-v * t * M_PI * M_PI))
           / (alfa + beta * cos(M_PI * x) * exp(-v * t * M_PI * M_PI));
}

void imprimeFuncao(char *nome, double x, double t, FPtr Uptr) {

    printf("\n  %s \n", nome);
    printf(" %10.6f ", Uptr(x, t));
}

/*
 * Cria nova matriz M[nLin][nCol] e inicializa com valor m
 * Entrada: nLin, nCol, **M, m
 */
void NewMatriz(int nLin, int nCol, double **&M, double m) {
    if (1) {//M == nullptr) {
        M = new double *[nLin];
        for (int i = 0; i < nLin; i++)
            M[i] = new double[nCol];

        for (int i = 0; i < nLin; i++)
            for (int j = 0; j < nCol; j++)
                M[i][j] = m;
    } else
        cout << "Atencao !! Matriz jah foi criada";
}

void NewVetor(int nCol, double *&V, double v) {
    if (1) {//V == nullptr) {
        V = new double[nCol];
        for (int i = 0; i < nCol; i++)
            V[i] = v;
    } else
        cout << "Atencao !! Vetor jah foi criado";
}

//void printArray(char ,double a[][N]) {
void imprimeMatriz(char *nome, int nlin, int ncol, double **&A) {
    printf("\n  %s \n", nome);
    // loop through array's rows
    for (int i = 0; i < nlin; ++i) {
        // loop through columns of current row
        for (int j = 0; j < ncol; ++j)
            printf(" %12.10f", A[i][j]);
        cout << endl; // start new line of output
    } // end outer for
} // end function printArray

//void printArray(char ,double a[][N]) {
void imprimeVetor(char *nome, int ncol, double *&v) {
    printf("\n  %s \n", nome);
    // loop through columns of current row
    for (int j = 0; j < ncol; ++j)
        printf(" %9.7f", v[j]);
    cout << endl; // start new line of output
} // end function printArray

void copiaVetor(int ini, int fim, double *&vOrig, double *&vDest) {
    for (int i = ini; i <= fim; i++)
        vDest[i] = vOrig[i];
}

/* Copia a coluna col da matriz MOrig no vetor vDest
 * for (int i = linIni; i <= linFim; i++)
 *       vDest[i] = MOrig[i][col];
 */
void copiaColMatrizToVetor(int col, int linIni, int linFim, double **&MOrig, double *&vDest) {
    for (int i = linIni; i <= linFim; i++)
        vDest[i] = MOrig[i][col];
}

void copiaVetorToColMatriz(int col, double *v, double **A) {
    for (int i = 0; i <= M; i++)
        A[i][col] = v[i];
}

/*  Preenche as três diagonais da matriz DF1
    DF1[0][0] = 1; // Condição de Dirichlet
    DF1[M][M] = 1; // Condição de Dirichlet
    DF1[0][1] = -sigma;
    DF1[M][M - 1] = -sigma;
    for (int i = 1; i < M; i++) {
        DF1[i][i] = 1 + 2 * sigma;
        DF1[i][i + 1] = -sigma;
        DF1[i][i - 1] = -sigma;
 */
void CalculaDF1(double **&DF1) {
    DF1[0][0] = 1; // Condição de Dirichlet
    DF1[M][M] = 1; // Condição de Dirichlet
    DF1[0][1] = -sigma;
    DF1[M][M - 1] = -sigma;
    for (int i = 1; i < M; i++) {
        DF1[i][i] = 1 + 2 * sigma;
        DF1[i][i + 1] = -sigma;
        DF1[i][i - 1] = -sigma;
    }
}

/* Preenche as três diagonais de DF2
   for (int i = 1; i < M; i++) {
        DF2[i][i] = (w1[i + 1] - w1[i - 1]) * k / (2 * h);
        DF2[i][i + 1] = w1[i] * k / (2 * h);
        DF2[i][i - 1] = -w1[i] * k / (2 * h);
    }
 */
void CalculaDF2(double **&DF2, double *&w1) {
    for (int i = 1; i < M; i++) {
        DF2[i][i] = (w1[i + 1] - w1[i - 1]) * k / (2 * h);
        DF2[i][i + 1] = w1[i] * k / (2 * h);
        DF2[i][i - 1] = -w1[i] * k / (2 * h);
    }
}

/*    Soma as matrizes
 *    S <-- A + B
 * */
void somaMatrizes(double **&S, double **&A, double **&B) {
    for (int i = 0; i <= M; i++)
        for (int j = 0; j <= M; j++)
            S[i][j] = A[i][j] + B[i][j];

}

/*
 * Multiplica a matriz A pelo escalar k
 *      A <-- kA
 * */
void escalar_x_matriz(double k, double **&A) {
    for (int i = 0; i <= M; i++)
        for (int j = 0; j <= M; j++)
            A[i][j] = A[i][j] * k;
}

/* Multiplicação da matriz A pelo vetor v
 *            vR <-- A*v
 * */
void Matriz_x_vetor(double *vR, double **A, double *v) {
    double soma;
    for (int i = 0; i <= M; i++) {
        soma = 0.0;
        for (int j = 0; j <= M; j++) {
            soma = soma + A[i][j] * v[j];
        }
        vR[i] = soma;
    }
}

/*     Usando o Lema 8.11, página 420  do livro
 *     Numerical Analysis, 2ªedition - Timothy Sauer
 *     for (int i = 0; i <= M; ++i)
 *         F[i] = -w[i][j] + wAux[i];
 * */
void CalculaF(int j, double *F, double **w, double *wAux) {
    for (int i = 0; i <= M; ++i)
        F[i] = -w[i][j] + wAux[i];

}

/* Condição de Dirichlet(da fronteira) para DF e F

    for (int j = 0; j <= M; j++) {
        DF[0][j] = 0.0;
        DF[M][j] = 0.0;
      }
      DF[0][0] = 1.0;
      DF[M][M] = 1.0;

      F[0] = w1[0] - EsqPtr(x0, n * k);
      F[M] = w1[M] - DirPtr(x1, n * k);

 *
 * */
void CondicaoDeDirichlet(int n, double **DF, double *F, double *w1, FPtr DirPtr, FPtr EsqPtr) {
    for (int j = 0; j <= M; j++) {
        DF[0][j] = 0.0;
        DF[M][j] = 0.0;
    }
    DF[0][0] = 1.0;
    DF[M][M] = 1.0;

    F[0] = w1[0] - EsqPtr(x0, n * k);
    F[M] = w1[M] - DirPtr(x1, n * k);
}

void ZeraMatriz(double **A) {
    for (int i = 0; i <= M; ++i) {
        for (int j = 0; j <= M; ++j) {
            A[i][j] = 0.0;

        }

    }
}