/*
    test m5b.c

 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

/* Set channel values to channel 4 */
#define ch00 0
#define ch01 1 << 8
#define ch10 2 << 8
#define ch11 3 << 8

int main() {
    size_t sz;
    __uint32_t w;
    __uint32_t *dat=NULL;
    size_t nar, idt, ifrm, ifrmdat;
    long qua[4] = {0, 0, 0, 0};  /* Four quantiles */
    long sum_qua = 0;
    __uint32_t ch_mask = 0x00000300; /* Set mask to channel 4 */
    int iqua;
    
    FILE *fh = NULL;
    
    printf("ch_mask = %08x\n", ch_mask);
    // return 0;
    
    fh = fopen("rd1910_wz_268-1811.m5b","rb");

    fseek(fh, 0L, SEEK_END);
    sz = ftell(fh);
    rewind(fh);

    printf("size = %d\n", sz);

    dat = (__uint32_t *) malloc(sz*sizeof(__uint32_t));

    /* Read the whole m5b file into RAM */
    fread(dat, sizeof(__uint32_t), sz, fh);

    ifrmdat = 4;
    for (ifrm=0; ifrm<3; ifrm++) {
    /*     printf("%05x", dat[idt] & 0xfffff);      // Whole seconds ot time */
    /*     printf(".%04x\n", dat[idt+1] / 0x10000); // Tenths of milliseconds */
        for (idt=0; idt<2500; idt++)
            switch (dat[ifrmdat+idt] & ch_mask) {
                case ch00: qua[0]++;
                    break;
                case ch01: qua[1]++;
                    break;
                case ch10: qua[2]++;
                    break;
                case ch11: qua[3]++;
                    break;
            }
        sum_qua = 0;
        for (iqua=0; iqua<4; iqua++) sum_qua += qua[iqua];
        
        printf("quantiles: ");
        for (iqua=0; iqua<4; iqua++) printf("%ld ", qua[iqua]);
        printf("  sum: %ld", sum_qua);
        printf("\n");

        for (iqua=0; iqua<4; iqua++) qua[iqua] = 0;
            
        ifrmdat = ifrmdat + 2504;
    }

    
    
   
    fclose(fh);

    // sleep(15);

    free(dat);
}






