#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"

float td[] = {
  0, 0, 0,
  1, 0, 1,
  0, 1, 1,
  1, 1, 0,
};

int main(){
    srand(time(NULL));

    size_t stride = 3;
    size_t n = sizeof(td)/sizeof(td[0])/stride;

    Mat ti = {
      .rows = n,
      .cols = 2,
      .stride = stride,
      .es = td,
    };

    Mat to = {
      .rows = n,
      .cols = stride-ti.cols,
      .stride = stride,
      .es = td+2,
    };

    size_t l[] = {2,2,1}; 
    size_t len = LEN(l);
    NN nn = nn_alloc(len, l);
    NN g = nn_alloc(len, l);
    nn_rand(nn, -1, 1);
    
    float eps = 1e-1;
    float rate = 1e-1;
    float epoch = 10000;
    for (size_t i = 0; i < epoch; i++) {
      nn_finete_diff(nn, g, eps, ti, to);
      nn_learn(nn, g, rate);
      float c = nn_cost(nn, ti, to);
      printf("%f\n", c);
    }

#if 1

    printf("-----------------------------\n");
  
    for (size_t i = 0; i <2; ++i) {
      for (size_t j = 0; j < 2; ++j) {
        MAT_AT(NN_INPUT(nn),0,0) = i;
        MAT_AT(NN_INPUT(nn),0,1) = j;
        nn_forward(nn);
        float y = *NN_OUTPUT(nn).es;
        printf("%zu | %zu = %f\n", i,j,y);
      }
    }
#endif /* if 0 */

    return 0;
}
