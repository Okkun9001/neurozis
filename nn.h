#ifndef NN_H_
#define NN_H_
#include <stddef.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // !NN_ASSERT

#define LEN(xs) sizeof(xs)/sizeof(xs[0])

// Matrix
typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  float* es;
} Mat;


Mat mat_alloc(size_t rows, size_t cols);
Mat mat_row(Mat a, size_t row);
void mat_copy(Mat r, Mat src);
void mat_rand(Mat m, float l, float h);
void mat_fill(Mat m, float a);
void mat_print(Mat m, const char *name, size_t padding);
#define MAT_AT(m,i,j) (m).es[(i)*(m).stride+(j)]
#define MAT_PRINT(m) mat_print(m, #m, 0)

void mat_dot(Mat r, Mat a, Mat b);
void mat_sum(Mat r, Mat a);

void mat_sig(Mat m);

// Neural Network
typedef struct {

  size_t count;
  Mat *ws;
  Mat *bs;
  Mat *as;
} NN;

NN nn_alloc(size_t len, size_t *l );
void nn_print(NN nn, const char *name);
void nn_rand(NN nn, float l, float h);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finete_diff(NN nn, NN g, float eps, Mat ti, Mat to);
void nn_learn(NN nn, NN g, float rate);
#define NN_PRINT(nn) nn_print(nn, #nn)
#define NN_INPUT(nn) nn.as[0]
#define NN_OUTPUT(nn) nn.as[nn.count]


// General
float rand_float(void);
float sigmoidf(float x);

#endif // !NN_H_


#ifdef NN_IMPLEMENTATION

#include <stdlib.h> // for malloc

Mat mat_alloc(size_t rows, size_t cols){
  Mat m;
  m.rows = rows;
  m.cols = cols;
  m.stride = cols;
  m.es = NN_MALLOC(sizeof(*m.es)*rows*cols);
  NN_ASSERT(m.es != NULL);
  return m;
}

void mat_dot(Mat r, Mat a, Mat b){

  NN_ASSERT(a.cols == b.rows);
  size_t n = a.cols;
  NN_ASSERT(r.rows == a.rows);
  NN_ASSERT(r.cols == b.cols);

  for (size_t i = 0; i < r.rows; i++) {
    for (size_t j = 0; j < r.cols; j++) {
      MAT_AT(r,i,j) = 0;
      for (size_t k = 0; k < n; k++) {
        MAT_AT(r,i,j) += MAT_AT(a,i,k)*MAT_AT(b,k,j);
      }
    }
  }
  (void) r;
  (void) a;
  (void) b;
}

void mat_sum(Mat r, Mat a){
  NN_ASSERT(r.rows == a.rows);
  NN_ASSERT(r.cols == a.cols);
  for(size_t i=0; i<a.rows; i++){
    for (size_t j = 0; j < a.cols; j++) {
      MAT_AT(r,i,j) += MAT_AT(a,i,j);
    }
  }
}

Mat mat_row(Mat m, size_t row){

  return (Mat){
    .rows=1,
    .cols=m.cols,
    .stride=m.stride,
    .es = &MAT_AT(m,row, 0)
  };
}


void mat_copy(Mat r, Mat src){
  NN_ASSERT(r.cols == src.cols);
  NN_ASSERT(r.rows == src.rows);
  for (size_t i = 0; i < r.rows; i++) {
    for (size_t j = 0; j < r.cols; j++) {
      MAT_AT(r,i,j) = MAT_AT(src,i,j);
    }
  }

}


void mat_print(Mat m, const char *name, size_t padding){
  printf("%*s%s = [\n",(int) padding, "", name);
  for (size_t i = 0; i<m.rows;i++) {
    printf("%*s   ",(int) padding, "");
    for (size_t j = 0; j < m.cols; j++) {
      printf("  %f ", MAT_AT(m,i,j));
    }
    printf("\n");
  }
  printf("%*s]\n", (int)padding, "");
}

void mat_rand(Mat m, float l, float h){
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m,i,j) = rand_float()*(h-l)+l;
    }
  }
}

float rand_float(void){
  return (float) rand() / (float) RAND_MAX;
}

void mat_fill(Mat m, float a){
  for (size_t i = 0; i < m.cols; i++) {
    for (size_t j = 0; j < m.rows; j++) {
      MAT_AT(m,i,j) = a;
    }
  }
}

float sigmoidf(float x){
  return 1.f/(1.f+expf(-x));
}

void mat_sig(Mat m){
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m,i,j) = sigmoidf(MAT_AT(m,i,j));
    }
  }
}

NN nn_alloc(size_t len, size_t *l ){
    NN nn;
    nn.count = len-1;
    nn.ws = NN_MALLOC(sizeof(*nn.ws)*nn.count);
    NN_ASSERT(nn.ws != NULL);
    nn.bs = NN_MALLOC(sizeof(*nn.bs)*nn.count);
    NN_ASSERT(nn.bs != NULL);
    nn.as = NN_MALLOC(sizeof(*nn.as)*len);
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(1, l[0]);
    for (size_t i = 0; i < nn.count; i++) {  
      nn.ws[i] = mat_alloc(l[i], l[i+1]);
      nn.bs[i] = mat_alloc(1, l[i+1]);
      nn.as[i+1] = mat_alloc(1,l[i+1]);
    }
    
    return nn;
}

void nn_print(NN nn, const char *name){

  char buf[256];
  printf("%s = [\n", name);
  for (size_t i = 0; i < nn.count; i++) {
    snprintf(buf, sizeof(buf), "ws%zu", i);
    mat_print(nn.ws[i], buf, 2);
    snprintf(buf, sizeof(buf), "bs%zu", i);
    mat_print(nn.bs[i], buf, 2);
  }
  printf("]\n");
}

void nn_rand(NN nn, float l, float h){
    for (size_t i = 0; i < nn.count; i++) {  
      mat_rand(nn.ws[i],l,h);
      mat_rand(nn.bs[i],l,h);
    }
}

void nn_forward(NN nn){

  size_t n = nn.count;

  for (size_t i = 0; i < n; i++) {
    mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
    mat_sum(nn.as[i+1], nn.bs[i]);
    mat_sig(nn.as[i+1]);
  }
}

float nn_cost(NN nn, Mat ti, Mat to){
  NN_ASSERT(ti.rows == to.rows);
  NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);

  size_t n = ti.rows;
  size_t q = to.cols;

  float c = 0;
  for (size_t i = 0; i < n; i++) {
    Mat x = mat_row(ti, i);
    Mat y = mat_row(to, i);
    mat_copy(NN_INPUT(nn), x);
    nn_forward(nn);

    for (size_t j = 0; j < q; j++) {
      float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
      c += d*d;
    }
  }

  return c/n;
}

void nn_finete_diff(NN nn, NN g, float eps, Mat ti, Mat to){
  
  size_t n = nn.count;
  float save;

  // Weight gradients 
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < nn.ws[i].rows; j++) {
      for (size_t k = 0; k < nn.ws[i].cols; k++) {
        save = MAT_AT(nn.ws[i], j, k);
        MAT_AT(nn.ws[i], j, k) -= eps;
        float cm = nn_cost(nn, ti, to);
        MAT_AT(nn.ws[i], j, k) += 2*eps;
        float cp = nn_cost(nn, ti, to);
        MAT_AT(g.ws[i],j,k) = (cp-cm)/(2*eps);
        MAT_AT(nn.ws[i], j, k) = save;
      }
    }
  }

  // Bias gradients 
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < nn.bs[i].rows; j++) {
      for (size_t k = 0; k < nn.bs[i].cols; k++) {
        save = MAT_AT(nn.bs[i], j, k);
        MAT_AT(nn.bs[i], j, k) -= eps;
        float cm = nn_cost(nn, ti, to);
        MAT_AT(nn.bs[i], j, k) += 2*eps;
        float cp = nn_cost(nn, ti, to);
        MAT_AT(g.bs[i],j,k) = (cp-cm)/(2*eps);
        MAT_AT(nn.bs[i], j, k) = save;
      }
    }
  }
}

void nn_learn(NN nn, NN g, float rate){

  for (size_t i = 0; i < nn.count; i++) {
    for (size_t j = 0; j < nn.ws[0].rows; j++) {
      for (size_t k = 0; k < nn.ws[0].cols; k++) {
        MAT_AT(nn.ws[i], j, k) -= MAT_AT(g.ws[i], j, k);
      }
    }
  }
  for (size_t i = 0; i < nn.count; i++) {
    for (size_t j = 0; j < nn.bs[0].rows; j++) {
      for (size_t k = 0; k < nn.bs[0].cols; k++) {
        MAT_AT(nn.bs[i], j, k) -= MAT_AT(g.bs[i], j, k);
      }
    }
  }
}
#endif // NN_IMPLEMENTATION
