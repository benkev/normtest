#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

const uint ND = 2;

uint bmul(uint a, uint b[], int nd);
uint badd(uint a, uint b[], int nd);

int main(int argc, char *argv[]) {
    int nd = ND;
    //uint bint[] = {65535, 0};
    uint bint[] = {65535, 0};
    uint a = 65535;
    ulong bint_id, res = 0;
    ulong abc, c;
    
    /*
     * Add bint = bint + a
     */

    printf("Add bint = bint + a:\n\n");
    
    bint[0] += a;

    for (int id=0; id<nd-1; id++) {
        bint_id = bint[id];
        if (bint_id > 65535) {
            bint[id+1] += bint_id / 65536;
            bint[id] =    bint_id % 65536;
        }
        printf("id = %d; bint[]: ", id);
        for (int id=nd-1; id>-1; id--) printf("%5u  ", bint[id]);
    printf("\n");
    }
    printf("\n\n");

    res = 0;
    res = 65536*bint[1] + bint[0];
    
    printf("bint[] = ");
    for (int id=nd-1; id>-1; id--) printf("%5u  ", bint[id]);
    printf("\n");
    printf("res = %lu\n", res);
    printf("\n");

    // return 0;

    /*
     * Multiply bint = a * bint
     */

    printf("==========================================\n");
    printf("Multiply bint = a * bint:\n\n");
    
    a = 93;
    bint[1] = 78; bint[0] = 28297;
    
    res = 65536*bint[1] + bint[0];
    printf("a = %d, b = %lu\n", a, res);
    

    c = 0; // Carry
    for (int id=0; id < nd; id++) {
        abc = a * bint[id] + c;
        bint[id] = abc % 65536;
        c =        abc / 65536;
        
        printf("id = %d; bint[] = ", id);
        for (int id=nd-1; id>-1; id--) printf("%5u  ", bint[id]);
        printf("\n");
    }
    printf("\n\n");

    res = 0;
    res = 65536*bint[1] + bint[0];
    
    printf("bint[] = ");
    for (int id=nd-1; id>-1; id--) printf("%5u  ", bint[id]);
    printf("\n");
    printf("c = %d, res = %lu\n", c, res);

    return 0;   
}


uint badd(uint a, uint b[], int nd) {

    uint c = 0; // Carry
    uint bc;   // = b + c

    b[0] += a;    
    for (int i = 0; i < nd; i++) {
        bc = b[i] + c;
        b[i] = bc % 65536;
        c =    bc / 65536;
    }
    
    return c;   
}

uint bmul(uint a, uint b[], int nd) {
    
    uint c = 0; // Carry
    uint abc;   // = a*b + c

    for (int i = 0; i < nd; i++) {
        abc = a * b[i] + c;
        b[i] = abc % 65536;
        c =    abc / 65536;
    }
    
    return c;   
}
