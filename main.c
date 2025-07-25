#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"

int main(){
    srand(time(NULL));
    Mat a = mat_alloc(2,2);
    mat_fill(a, 2);
    float id_data[4] = {1,0,0,1}; 
    Mat b = { .rows=2, .cols=2, .es=id_data};
    Mat r = mat_alloc(2,2);
    mat_dot(r, a, b);
    mat_print(r);
    return 0;
}
