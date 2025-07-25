#ifndef NN_H_
#define NN_H_
#include <stddef.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // !NN_ASSERT


typedef struct {
  size_t rows;
  size_t cols;
  float* es;
} Mat;

#define MAT_AT(m,i,j) (m).es[(i)*(m).cols+(j)]

float rand_float(void);

Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float l, float h);
void mat_fill(Mat m, float a);
void mat_print(Mat m);

void mat_dot(Mat r, Mat a, Mat b);
void mat_sum(Mat r, Mat a);


#endif // !NN_H_


#ifdef NN_IMPLEMENTATION

#include <stdlib.h> // for malloc

Mat mat_alloc(size_t rows, size_t cols){
  Mat m;
  m.rows = rows;
  m.cols = cols;
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
  NN_ASSERT(r.cols == r.rows);
  for(size_t i=0; i<a.rows; i++){
    for (size_t j = 0; j < a.cols; j++) {
      MAT_AT(r,i,j) += MAT_AT(a,i,j);
    }
  }
}

void mat_print(Mat m){
  for (size_t i = 0; i<m.rows;i++) {
    for (size_t j = 0; j < m.cols; j++) {
      printf("%f ", MAT_AT(m,i,j));
    }
    printf("\n");
  }
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

#endif // NN_IMPLEMENTATION
