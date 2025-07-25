all:
	gcc -Wall -Wextra main.c -o neurozis

run: all
	./neurozis
