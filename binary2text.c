//  Copyright 2016 Eduardo A. Brito Chacón. All Rights Reserved.
//
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
#include <malloc.h>




const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries


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

int main(int argc, char **argv) {
  FILE *f,*fo1,*fo2,*fo3;
  char file_name[max_size],file_name1[max_size],file_name2[max_size],file_name3[max_size];
  long long words, size, a,b;
  float *Mu,*Sigma;
  char *vocab;
  int i, sigma=1, header=1, bin=0, sep_mat =0;
  if (argc < 2) {
    printf("Usage: ./binary2text <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    printf("\t-sim <SIM>\n");
    printf("\twhere SIM sets the similarity measure: Expected Likelihood or Cosine Distance (el or cos); default is el\n");
    printf("\t-header <int>\n");
    printf("\t 0 drops the header of the resulting file; default is 1\n");
    printf("\t-sigma <int>\n");
    printf("\t 0 drops the covariance matrix of the embeddings; default is 1\n");
    printf("\t-sep-mat <int>\n");
    printf("\t 1 generates different files for the mean and for the covariance matrix; default is 0\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  if ((i = ArgPos((char *)"-sigma", argc, argv)) > 0) 
    sigma = strcmp("0",argv[i + 1]);
  if ((i = ArgPos((char *)"-header", argc, argv)) > 0) 
    header = strcmp("0",argv[i + 1]);
  if ((i = ArgPos((char *)"-bin", argc, argv)) > 0) 
    bin = !strcmp("1",argv[i + 1]);
  if ((i = ArgPos((char *)"-sep-mat", argc, argv)) > 0) 
    sep_mat = !strcmp("1",argv[i + 1]);
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
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
    for (a = 0; a < size; a++)  fread(&Mu[a + b * size], sizeof(float), 1, f);
    for (a = 0; a < size; a++)  fread(&Sigma[a + b * size], sizeof(float), 1, f);
  }
  fclose(f);
  strcpy(file_name1 ,file_name);
  strcpy(file_name2 ,file_name);
  strcpy(file_name3 ,file_name); 
  if (sep_mat){
    strcat(file_name1, "_mu.txt");
    strcat(file_name2, "_sigma.txt");
    strcat(file_name3, "_words.txt");
    fo2 = fopen(file_name2, "wb");
    fo3 = fopen(file_name3, "wb");
  }
  else  strcat(file_name1, ".txt");  
  fo1 = fopen(file_name1, "wb");


  if (header)
    fprintf(fo1, "%lld %lld\n", words, size);
  for (a = 0; a < words; a++) {
    if (sep_mat) fprintf(fo3, "%s\n", &vocab[a*max_w]);
    else  fprintf(fo1, "%s ", &vocab[a*max_w]);
    for (b = 0; b < size; b++){
      if (bin)
          fwrite(&Mu[a*size + b], sizeof(float), 1, fo1); 
      else
          fprintf(fo1, "%lf ", Mu[a*size + b]);
    }
    if (sigma){
      FILE *fo;
      if (sep_mat)
        fo = fo2;
      else
        fo = fo1;
      for (b = 0; b < size; b++) 
        if (bin)
            fwrite(&Sigma[a*size + b], sizeof(float), 1, fo);
        else
            fprintf(fo, "%lf ", Sigma[a*size + b]);
      fprintf(fo2, "\n");
    }
    fprintf(fo1, "\n");
  } 
  fclose(fo1);
  if (sep_mat){
    fclose(fo2);
    fclose(fo3);
  }
  return 0;
}
