/*
 * C Macro Definition on Command Line vs in Code 
 *
 * On the command line with GCC, -DFOO is a shorthand for -DFOO=1.
 *
 * To demonstrate:
 *
 * $ gcc -o md -DFOO define_in_cmdline.c && ./md
 * Foo 1
 * $
 *
 * This is actually required by the POSIX standard for c99:
 *
 * -D  name[=value]
 * Define name as if by a C-language #define directive.
 * If no =value is given, a value of 1 shall be used.
 * The -D option has lower precedence than the -U option. That is, if name
 * is used in both a -U and a -D option, name shall be undefined regardless
 * of the order of the options.
 */

#include <stdio.h>

// #define FOO

int main() {
#ifdef FOO
    printf("Foo %d\n", FOO);
#else
    printf("Bar %d\n", 9876);
#endif
} 
