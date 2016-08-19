#CC = gcc
CC = gcc -g
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: learn distance binary2text specificity

learn : learn.c
	$(CC) learn.c -o learn $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o distance $(CFLAGS)
specificity : specificity.c
	$(CC) specificity.c -o specificity $(CFLAGS)
binary2text: binary2text.c
	$(CC) binary2text.c -o binary2text $(CFLAGS)
clean:
	rm -rf learn distance binary2text specificity