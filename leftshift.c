#include <stdio.h>
#include <stdint.h>

int main() {
    __uint32_t msk = 0x03, i;

    for (i=0; i<16; i+=2) {
        printf("%02d  %08x\n", i, msk << i);
    }
}



