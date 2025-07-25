all:
	@gcc -Wall -Wextra main.c -o neurozis -lm

run: all
	@./neurozis
