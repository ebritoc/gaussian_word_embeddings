//  Copyright 2016 Eduardo A. Brito Chacón. All Rights Reserved.
//  Derivated work from word2vec
//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>

#define COST_CONST 0.398942280401
#define PI 3.14159265358979

const long long max_size = 2000;         // max length of strings
const long long N = 100;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries
const long long max_distance = 10e6;

struct Stats {
   char  *word;
   float  distance;
   float  variance;
};

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

//Calculates the logarithm of the variance
float diagonal_variance(float *sigma, int size){
  //float det=1;
  float det=0;
  int i;
  //for (i = 0; i < size; i++)  det *= sigma[i];
  for (i = 0; i < size; i++)  det += log(sigma[i]);
  return det;  
}

//Compare function to use with qsort
int cmpfunc (const void * a, const void * b){
  return ( (*(struct Stats*) a).variance > (*(struct Stats*) b).variance );
}

int main(int argc, char **argv) {
  FILE *f;
  char st1[max_size];
  char *bestw[N];
  char file_name[max_size], st[100][max_size];
  float dist, bestd[N], vec[max_size],len, bestvar[N];
  long long words, size, a, b, c, d, cn, bi[100];
  float *Mu,*Sigma,loss_const;
  char *vocab;
  int i, cos=0;
  struct Stats topN[N];
  if (argc < 2) {
    printf("Usage: ./distance <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
   if ((i = ArgPos((char *)"-sim", argc, argv)) > 0) cos = (!strcmp("cos",argv[i + 1]));
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  loss_const = pow(2*PI,  (float) size / 2);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
  Mu = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (Mu == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  Sigma = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (Sigma == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  } 
  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }

    vocab[b * max_w + a] = 0;
    for (a = 0; a < size; a++)	fread(&Mu[a + b * size], sizeof(float), 1, f);
    for (a = 0; a < size; a++)  fread(&Sigma[a + b * size], sizeof(float), 1, f);

    if(cos){ //Cosine distance between means: normalization
    len = 0;
    for (a = 0; a < size; a++) len += Mu[a + b * size] * Mu[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) Mu[a + b * size] /= len;
    } 
       
    
  }
  fclose(f);
  while (1) {
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    for (a = 0; a < N; a++) bestvar[a] = 0;

    printf("Enter word or sentence (EXIT to break): ");
    a = 0;
    while (1) {
      st1[a] = fgetc(stdin);
      if ((st1[a] == '\n') || (a >= max_size - 1)) {
        st1[a] = 0;
        break;
      }
      a++;
    }
    if (!strcmp(st1, "EXIT")) break;
    cn = 0;
    b = 0;
    c = 0;
    while (1) {
      st[cn][b] = st1[c];
      b++;
      c++;
      st[cn][b] = 0;
      if (st1[c] == 0) break;
      if (st1[c] == ' ') {
        cn++;
        b = 0;
        c++;
      }
    }
    cn++;
    for (a = 0; a < cn; a++) {
      for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st[a])) break;
      if (b == words) b = -1;
      bi[a] = b;
      printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
      if (b == -1) {
        printf("Out of dictionary word!\n");
        break;
      }
    }
    if (b == -1) continue;
    printf("\n                Word\tVariance\t\t\tSimilarity\n------------------------------------------------------------------------\n");
    for (a = 0; a < 2*size; a++) vec[a] = 0;
    for (b = 0; b < cn; b++) {
      if (bi[b] == -1) continue;
     // for (a = 0; a < size; a++) vec[a] += Mu[a + bi[b] * size];
       for (a = 0; a < size; a++){
        vec[a] += Mu[a + bi[b] * size];
        vec[a+size] += Sigma[a + bi[b] * size];
       } 
    }
    if (cos){
      len = 0;
      for (a = 0; a < size; a++) len += vec[a] * vec[a];
      len = sqrt(len);
      for (a = 0; a < size; a++) vec[a] /= len;
      for (a = 0; a < N; a++) bestd[a] = max_distance;  
    }

    
    for (a = 0; a < N; a++) bestd[a] = -max_distance;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    for (c = 0; c < words; c++) {
      a = 0;
      for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
      if (a == 1) continue;
      dist = 0;
      for (a = 0; a < size; a++){
      	//dist += (vec[a] - Mu[a + c * size])*(vec[a] - Mu[a + c * size]) / Sigma[a + c * size];
        //dist += (vec[a] - Mu[a + c * size])*(vec[a] - Mu[a + c * size]) / vec[a+size];
        //dist += log((pow(COST_CONST, size) / sqrt(vec[a+size] + Sigma[a + c * size]))
        //  *exp(-0.5*(Mu[a + c * size] - vec[a])*(Mu[a + c * size] - vec[a]) / (vec[a+size] + Sigma[a + c * size])   )) ;
        if (cos) //Cosine distance between means
          dist += vec[a] * Mu[a + c * size];        
        else //Expected likelihood similarity
                dist += (-0.5*(Mu[a + c * size] - vec[a])*(Mu[a + c * size] - vec[a]) / (vec[a+size] + Sigma[a + c * size])   )
                                - log(sqrt(vec[a+size] + Sigma[a + c * size]) * loss_const); 
      } 
      //dist = sqrt(dist);
      for (a = 0; a < N; a++) {
        if (dist > bestd[a]) {        
          for (d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            bestvar[d] = bestvar[d - 1];
            strcpy(bestw[d], bestw[d - 1]);
          }
          bestd[a] = dist;
          bestvar[a] = diagonal_variance(Sigma + c * size, size);
          strcpy(bestw[a], &vocab[c * max_w]);
          break;
        }
      }
    }
    //for (a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);
    //Calculate the stats for the top N words
    for (a = 0; a < N; a++){
      topN[a].word = bestw[a];
      topN[a].distance = bestd[a];
      topN[a].variance = bestvar[a];
    } 
    qsort(&topN, N, sizeof(struct Stats), cmpfunc);
    for (a = N-1; a >=0 ; a--) printf("%20s\t%f\t\t\t%f\n", topN[a].word, topN[a].variance , topN[a].distance);
  }
  return 0;
}
