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
#include <math.h>
#include <pthread.h>
#include <fenv.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define COST_CONST 0.398942280401
#define RIDGE 1e-8

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1, batch_size =20, kl_energy=0, el_energy=1;
int *vocab_hash;
int vocab_max_size = 1000, vocab_size = 0, layer1_size = 200, vector_size=100, variance_size=100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0;
real alpha = 1, starting_alpha, sample = 1e-3, epsilon=0.1;
real *syn0, *syn1, *syn1neg, *expTable, *grad_acc0,*grad_acc1;

clock_t start;

real C = 1, m=0.1, M=10, loss_margin=1;
const int table_size = 1e8;
int *table;

//Returns the l2 norm of the given vector
real l2_norm(real *v){
  real squares = 0;
  int i;
  for (i = 0; i < vector_size; i++)
    squares += v[i]*v[i];
  return sqrt(squares);
}

//Regularizes a mean vector in case its norms exceeds C
void regularize_mu(real *mu){
 real norm = l2_norm(mu);
          if (norm > C){
            int i;
            for (i = 0; i < vector_size; i++)
              mu[i] *= C/norm;
          }
}


real loss(real *w, real *cp, real *cn){
  real ep=0, en =0, res;
  int i;
  if (el_energy) //Constants of the energy function dropped. 0.5 factor applied at the end with res
  	for (i = 0; i < vector_size; i++){
 		ep -= (w[i] - cp[i])*(w[i] - cp[i]) / (w[vector_size + i] + cp[vector_size + i])   
  			+ log(w[vector_size + i] + cp[vector_size + i]);
  		en -= (w[i] - cn[i])*(w[i] - cn[i]) / (w[vector_size + i] + cn[vector_size + i])   
  			+ log(w[vector_size + i] + cn[vector_size + i]);  		
  	}
  else //KL divergence
  	for (i = 0; i < vector_size; i++){ //log(w[vector_size + i]) removed from both lines since they cancel
  		ep -= cp[vector_size + i] / w[vector_size + i] + (w[i] - cp[i])*(w[i] - cp[i])*w[vector_size + i]
  			- log(cp[vector_size + i]);
  		en -= cn[vector_size + i] / w[vector_size + i] + (w[i] - cn[i])*(w[i] - cn[i])*w[vector_size + i]
  			- log(cn[vector_size + i]);
  	}
  res = (loss_margin - 0.5*(ep - en));
  
  if (res > 0)
    return res;
  else 
    return 0;
}

void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));  
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}



void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %d\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %d\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&syn0, 128, vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  a = posix_memalign((void **)&syn1, 128, vocab_size * layer1_size * sizeof(real));
  if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  a = posix_memalign((void **)&grad_acc0, 128, vocab_size * layer1_size * sizeof(real));
  if (grad_acc0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  a = posix_memalign((void **)&grad_acc1, 128, vocab_size * layer1_size * sizeof(real));
  if (grad_acc1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  
  

    for (a = 0; a < vocab_size; a++) for (b = 0; b < vector_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = 2*C* (((next_random & 0xFFFF) / (real)65536) - 0.5) / vector_size ;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + vector_size + b] = m + (M-m)*((next_random & 0xFFFF) / (real)65536) ;

    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn1[a * layer1_size + b] = 2*C* (((next_random & 0xFFFF) / (real)65536) - 0.5) / vector_size  ;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn1[a * layer1_size + vector_size + b] = m + (M-m)*((next_random & 0xFFFF) / (real)65536);
    }
  memset(grad_acc0, 0, vocab_size * layer1_size * sizeof(real));
  memset(grad_acc1, 0, vocab_size * layer1_size * sizeof(real));

}

void UpdateWeights(real *syn0_g, real *syn1_g){
  long long a,b;    
  for (a = 0; a < vocab_size; a++){
  	if (syn0_g[a*layer1_size] || syn0_g[a*layer1_size + 1]){ //Workaround to check if the respective word was updated
      //Mu update for hidden layer 
      for (b = 0; b < vector_size; b++){        
        //Clip to maximum possible update   
        if (syn0_g[a*layer1_size + b] > 2*C)
          syn0_g[a*layer1_size + b]  = 2*C;
        if (syn0_g[a*layer1_size + b] < -2*C)
          syn0_g[a*layer1_size + b] = -2*C;

        // Update of the weight
        syn0[a*layer1_size + b] -= syn0_g[a*layer1_size + b] * alpha / sqrt(grad_acc0[a*layer1_size + b] + epsilon )  ;
        //Accumulation of historical gradients
        grad_acc0[a*layer1_size + b] += syn0_g[a*layer1_size + b]*syn0_g[a*layer1_size + b];

      }
      regularize_mu(&syn0[a * layer1_size]);  

      //Sigma update for hidden layer
      for (b = 0; b < variance_size; b++){      
        //Clip sigma updates
        if (syn0_g[a*layer1_size + vector_size + b]  > M)
          syn0_g[a*layer1_size + vector_size + b]  = M;
        if (syn0_g[a*layer1_size + vector_size + b]  < -M)
          syn0_g[a*layer1_size + vector_size + b]  = -M;
		
		syn0[a*layer1_size + vector_size + b] -= syn0_g[a*layer1_size + vector_size + b] * alpha / sqrt(grad_acc0[a*layer1_size + vector_size + b] + epsilon ) ;
        grad_acc0[a*layer1_size + vector_size + b] += syn0_g[a*layer1_size + vector_size + b] * syn0_g[a*layer1_size + vector_size + b];
        
      
        //Regularization of sigma
        if (syn0[a*layer1_size + vector_size + b] > M )
          syn0[a*layer1_size + vector_size + b] = M;
        else
          if ( syn0[a*layer1_size + vector_size + b] < m)
            syn0[a*layer1_size + vector_size + b] = m;
      }
  	} 
  	if (syn1_g[a*layer1_size] || syn1_g[a*layer1_size + 1] ){ //Workaround to check whether the word was updated   
      //Mu update for output layer 
      for (b = 0; b < vector_size; b++){
        //Clip to maximum possible update   
        if (syn1_g[a*layer1_size + b] > 2*C)
          syn1_g[a*layer1_size + b]  = 2*C;
        if (syn1_g[a*layer1_size + b] < -2*C)
          syn1_g[a*layer1_size + b] = -2*C;

      	// Update of the weight
        syn1[a*layer1_size + b] -= syn1_g[a*layer1_size + b] * alpha / sqrt(grad_acc1[a*layer1_size + b] + epsilon )  ;
        //Accumulation of historical gradients
        grad_acc1[a*layer1_size + b] += syn1_g[a*layer1_size + b]*syn1_g[a*layer1_size + b];
        
      }
      regularize_mu(&syn1[a * layer1_size]);  

      //Sigma update for output layer
      for (b = 0; b < variance_size; b++){
        //Clip sigma updates
        if (syn1_g[a*layer1_size + vector_size + b]  > M)
          syn1_g[a*layer1_size + vector_size + b]  = M;
        if (syn1_g[a*layer1_size + vector_size + b]  < -M)
          syn1_g[a*layer1_size + vector_size + b]  = -M;

        syn1[a*layer1_size + vector_size + b] -= syn1_g[a*layer1_size + vector_size + b] * alpha / sqrt(grad_acc1[a*layer1_size + vector_size + b] + epsilon ) ;
        grad_acc1[a*layer1_size + vector_size + b] += syn1_g[a*layer1_size + vector_size + b] * syn1_g[a*layer1_size + vector_size + b];       
      
        //Regularization of sigma
        if (syn1[a*layer1_size + vector_size + b] > M )
          syn1[a*layer1_size + vector_size + b] = M;
        else
          if ( syn1[a*layer1_size + vector_size + b] < m)
            syn1[a*layer1_size + vector_size + b] = m;
      }
  	}
    
  }
  
  memset(syn0_g, 0, vocab_size * layer1_size * sizeof(real));
  memset(syn1_g, 0, vocab_size * layer1_size * sizeof(real));  
}

void *TrainModelThread(void *id) {
  long long a, b, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, c, target, local_iter = iter;
  long long sentence_i = 0;
  unsigned long long next_random = (long long)id;
  real current_cost=-1;
  clock_t now;
  real *syn0_g, *syn1_g; 
  
  // Initialization of the local gradients
  posix_memalign((void **)&syn0_g, 128, vocab_size * layer1_size * sizeof(real));
  if (syn0_g == NULL) {printf("Memory allocation failed\n"); exit(1);}
  posix_memalign((void **)&syn1_g, 128, vocab_size * layer1_size * sizeof(real));
  if (syn1_g == NULL) {printf("Memory allocation failed\n"); exit(1);}
  memset(syn0_g, 0, vocab_size * layer1_size * sizeof(real));
  memset(syn1_g, 0, vocab_size * layer1_size * sizeof(real));


  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cProgress: %.2f%%  Words/thread/sec: %.2fk  \t Cost function:%f", 13,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000),
         current_cost);
        fflush(stdout);
      }
     
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);

        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;

        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / num_threads * (long long)id, SEEK_SET);
      continue;
    }

    word = sen[sentence_position];
    if (word == -1) continue;


    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        long long cp, cn;
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
                           
        //target = word;            
        cp = word * layer1_size; //Positive context word
        {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            cn = target * layer1_size; // For cost function
         }
        // printf("cp %d cn %d \n", cp,cn );
        if ( (current_cost = loss(&syn0[l1], &syn1[cp], &syn1[cn]))){
        //  update_count++;
          for (c = 0; c < vector_size; c++){
          	real g_sigma_p, g_sigma_n, g_sigma_w, delta_p, delta_n, denom_p, denom_n, sigma_w_inv,sigma_p_inv,sigma_n_inv;
          	if (el_energy){
          		denom_p = syn0[l1 + vector_size +c] + syn1[cp + vector_size + c] + RIDGE;
           		delta_p = (syn0[l1 + c] - syn1[cp + c]) /  denom_p;
           		g_sigma_p = -0.5*(delta_p*delta_p - 1/denom_p) ;
           		denom_n = syn0[l1 + vector_size +c] + syn1[cn + vector_size + c] + RIDGE;
           		delta_n = (syn0[l1 + c] - syn1[cn + c]) /  denom_n;
           		g_sigma_n = 0.5*(delta_n*delta_n - 1 /denom_n) ;  
  				//Covariance gradients 
           		g_sigma_w = g_sigma_n + g_sigma_p;
          	}
          	else{ //KL energy
          		sigma_w_inv = 1/syn0[l1 + vector_size + c];
          		sigma_p_inv = 1/syn1[cp + vector_size + c];
          		sigma_n_inv = 1/syn1[cn + vector_size + c];
          		delta_p = sigma_w_inv * (syn0[l1 + c] - syn1[cp + c]);
          		delta_n = sigma_w_inv * (syn0[l1 + c] - syn1[cn + c]);

          		g_sigma_p = -0.5 * ( sigma_p_inv - sigma_w_inv);
          		g_sigma_n =  0.5 * ( sigma_n_inv - sigma_w_inv);
          		g_sigma_w = 0.5 * (sigma_w_inv*sigma_w_inv*(syn1[cn + vector_size + c]  - syn1[cp + vector_size + c]) + delta_p*delta_p - delta_n*delta_n );

           		syn0_g[l1 + c + vector_size] += g_sigma_n + g_sigma_p;			
  				syn1_g[c + vector_size+ cp] += g_sigma_p;
				syn1_g[c + vector_size + cn] += g_sigma_n;
          	}            

			//Mu gradients
            syn0_g[l1 + c] += delta_p - delta_n;          	
            syn1_g[c + cp] -= delta_p;            
            syn1_g[c + cn] += delta_n ;
            //Covariance gradients 
           	syn0_g[l1 + c + vector_size] += g_sigma_w;			
  			syn1_g[c + vector_size + cp] += g_sigma_p;
			syn1_g[c + vector_size + cn] += g_sigma_n;	
            
          }      
        }
    } 
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      //Update weights in case the batch is finished      
      if ( (++sentence_i % batch_size) == 0)   UpdateWeights(syn0_g, syn1_g);      
      continue;
    }
  }
  
  UpdateWeights(syn0_g, syn1_g);; //last weight updates for the last batch
  free(syn0_g);
  free(syn1_g);
  fclose(fi);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  InitUnigramTable();
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo = fopen(output_file, "wb");
  // Save the word vectors
  fprintf(fo, "%d %d\n", vocab_size, vector_size);
  for (a = 0; a < vocab_size; a++) {
    fprintf(fo, "%s ", vocab[a].word);
    if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
    else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
    fprintf(fo, "\n");
	}
  fclose(fo);
}

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
  int i;

  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 1\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-C <int>\n");
    printf("\t\tMax l2 norm of mean vectors; default is 1\n");
    printf("\t-m <int>\n");
    printf("\t\tMin covariance eigenvalue; default is 0.1\n");
    printf("\t-M <int>\n");
    printf("\t\tMax covariance eigenvalue; default is 0.1\n");
    printf("\t-loss-margin <int>\n");
    printf("\t\tLoss function margin; default is 0\n");
    printf("\t-batch <int>\n");
    printf("\t\tBatch size; default is 20\n");
    printf("\t-energy <str>\n");
    printf("\t\tEnergy function (el or kl); default is el\n");
    printf("\t-epsilon <int>\n");
    printf("\t\tRidge added in the adagrad update within sqrt; default is 0.1\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -binary 0 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0){vector_size = atoi(argv[i + 1]);  layer1_size = 2*vector_size; variance_size = vector_size;} // Mean + diagonal covariance matrix
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-C", argc, argv)) > 0) C = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-M", argc, argv)) > 0) M = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-m", argc, argv)) > 0) m = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-loss-margin", argc, argv)) > 0) loss_margin = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-batch", argc, argv)) > 0) batch_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-energy", argc, argv)) > 0){kl_energy = strcmp(argv[i + 1], "kl"  ); el_energy = !el_energy;}
  if ((i = ArgPos((char *)"-epsilon", argc, argv)) > 0) epsilon = atof(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
