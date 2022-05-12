#include <stdio.h>

void fred();
void barbara ( void (*function_ptr)() );
int main();

int main() {
  barbara (fred);
  return 0;
}

void fred() {
  printf("fred here!\n");
}

void barbara ( void (*function_ptr)() ) {
  /* Call fred */
  (*function_ptr)();
}
 
