/*
 * Copyright 2011-2020 The OpenSSL Project Authors. All Rights Reserved.
 *
 * Licensed under the OpenSSL license (the "License").  You may not use
 * this file except in compliance with the License.  You can obtain a copy
 * in the file LICENSE in the source distribution or at
 * https://www.openssl.org/source/license.html
 */

#include <stdlib.h>
#include <string.h>
#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/rand.h>
#include "modes_local.h"
#include "internal/thread_once.h"
#include "rand_local.h"
#include "sanitizers.h"

/*
 * Implementation of NIST SP 800-90A CTR DRBG.
 */

static void inc_128(RAND_DRBG_CTR *ctr)
{
    unsigned char *p = &ctr->V[0];
    u32 n = 16, c = 1;

    do {
        --n;
        c += p[n];
        p[n] = (u8)c;
        c >>= 8;
    } while (n);
}

static void ctr_XOR(RAND_DRBG_CTR *ctr, const unsigned char *in, size_t inlen)
{
    size_t i, n;

    if (in == NULL || inlen == 0)
        return;

    /*
     * Any zero padding will have no effect on the result as we
     * are XORing. So just process however much input we have.
     */
    n = inlen < ctr->keylen ? inlen : ctr->keylen;
    for (i = 0; i < n; i++)
        ctr->K[i] ^= in[i];
    if (inlen <= ctr->keylen)
        return;

    n = inlen - ctr->keylen;
    if (n > 16) {
        /* Should never happen */
        n = 16;
    }
    for (i = 0; i < n; i++)
        ctr->V[i] ^= in[i + ctr->keylen];
}

/*
 * Process a complete block using BCC algorithm of SP 800-90A 10.3.3
 */
__owur static int ctr_BCC_block(RAND_DRBG_CTR *ctr, unsigned char *out,
                                const unsigned char *in, int len)
{
    int i, outlen = AES_BLOCK_SIZE;

    for (i = 0; i < len; i++)
        out[i] ^= in[i];

    if (!EVP_CipherUpdate(ctr->ctx_df, out, &outlen, out, len)
        || outlen != len)
        return 0;
    return 1;
}


/*
 * Handle several BCC operations for as much data as we need for K and X
 */
__owur static int ctr_BCC_blocks(RAND_DRBG_CTR *ctr, const unsigned char *in)
{
    unsigned char in_tmp[48];
    unsigned char num_of_blk = 2;

    memcpy(in_tmp, in, 16);
    memcpy(in_tmp + 16, in, 16);
    if (ctr->keylen != 16) {
        memcpy(in_tmp + 32, in, 16);
        num_of_blk = 3;
    }
    return ctr_BCC_block(ctr, ctr->KX, in_tmp, AES_BLOCK_SIZE * num_of_blk);
}

/*
 * Initialise BCC blocks: these have the value 0,1,2 in leftmost positions:
 * see 10.3.1 stage 7.
 */
__owur static int ctr_BCC_init(RAND_DRBG_CTR *ctr)
{
    unsigned char bltmp[48] = {0};
    unsigned char num_of_blk;

    memset(ctr->KX, 0, 48);
    num_of_blk = ctr->keylen == 16 ? 2 : 3;
    bltmp[(AES_BLOCK_SIZE * 1) + 3] = 1;
    bltmp[(AES_BLOCK_SIZE * 2) + 3] = 2;
    return ctr_BCC_block(ctr, ctr->KX, bltmp, num_of_blk * AES_BLOCK_SIZE);
}

/*
 * Process several blocks into BCC algorithm, some possibly partial
 */
__owur static int ctr_BCC_update(RAND_DRBG_CTR *ctr,
                                 const unsigned char *in, size_t inlen)
{
    if (in == NULL || inlen == 0)
        return 1;

    /* If we have partial block handle it first */
    if (ctr->bltmp_pos) {
        size_t left = 16 - ctr->bltmp_pos;

        /* If we now have a complete block process it */
        if (inlen >= left) {
            memcpy(ctr->bltmp + ctr->bltmp_pos, in, left);
            if (!ctr_BCC_blocks(ctr, ctr->bltmp))
                return 0;
            ctr->bltmp_pos = 0;
            inlen -= left;
            in += left;
        }
    }

    /* Process zero or more complete blocks */
    for (; inlen >= 16; in += 16, inlen -= 16) {
        if (!ctr_BCC_blocks(ctr, in))
            return 0;
    }

    /* Copy any remaining partial block to the temporary buffer */
    if (inlen > 0) {
        memcpy(ctr->bltmp + ctr->bltmp_pos, in, inlen);
        ctr->bltmp_pos += inlen;
    }
    return 1;
}

__owur static int ctr_BCC_final(RAND_DRBG_CTR *ctr)
{
    if (ctr->bltmp_pos) {
        memset(ctr->bltmp + ctr->bltmp_pos, 0, 16 - ctr->bltmp_pos);
        if (!ctr_BCC_blocks(ctr, ctr->bltmp))
            return 0;
    }
    return 1;
}

__owur static int ctr_df(RAND_DRBG_CTR *ctr,
                         const unsigned char *in1, size_t in1len,
                         const unsigned char *in2, size_t in2len,
                         const unsigned char *in3, size_t in3len)
{
    static unsigned char c80 = 0x80;
    size_t inlen;
    unsigned char *p = ctr->bltmp;
    int outlen = AES_BLOCK_SIZE;

    if (!ctr_BCC_init(ctr))
        return 0;
    if (in1 == NULL)
        in1len = 0;
    if (in2 == NULL)
        in2len = 0;
    if (in3 == NULL)
        in3len = 0;
    inlen = in1len + in2len + in3len;
    /* Initialise L||N in temporary block */
    *p++ = (inlen >> 24) & 0xff;
    *p++ = (inlen >> 16) & 0xff;
    *p++ = (inlen >> 8) & 0xff;
    *p++ = inlen & 0xff;

    /* NB keylen is at most 32 bytes */
    *p++ = 0;
    *p++ = 0;
    *p++ = 0;
    *p = (unsigned char)((ctr->keylen + 16) & 0xff);
    ctr->bltmp_pos = 8;
    if (!ctr_BCC_update(ctr, in1, in1len)
        || !ctr_BCC_update(ctr, in2, in2len)
        || !ctr_BCC_update(ctr, in3, in3len)
        || !ctr_BCC_update(ctr, &c80, 1)
        || !ctr_BCC_final(ctr))
        return 0;
    /* Set up key K */
    if (!EVP_CipherInit_ex(ctr->ctx_ecb, NULL, NULL, ctr->KX, NULL, -1))
        return 0;
    /* X follows key K */
    if (!EVP_CipherUpdate(ctr->ctx_ecb, ctr->KX, &outlen, ctr->KX + ctr->keylen,
                          AES_BLOCK_SIZE)
        || outlen != AES_BLOCK_SIZE)
        return 0;
    if (!EVP_CipherUpdate(ctr->ctx_ecb, ctr->KX + 16, &outlen, ctr->KX,
                          AES_BLOCK_SIZE)
        || outlen != AES_BLOCK_SIZE)
        return 0;
    if (ctr->keylen != 16)
        if (!EVP_CipherUpdate(ctr->ctx_ecb, ctr->KX + 32, &outlen,
                              ctr->KX + 16, AES_BLOCK_SIZE)
            || outlen != AES_BLOCK_SIZE)
            return 0;
    return 1;
}

/*
 * NB the no-df Update in SP800-90A specifies a constant input length
 * of seedlen, however other uses of this algorithm pad the input with
 * zeroes if necessary and have up to two parameters XORed together,
 * so we handle both cases in this function instead.
 */
__owur static int ctr_update(RAND_DRBG *drbg,
                             const unsigned char *in1, size_t in1len,
                             const unsigned char *in2, size_t in2len,
                             const unsigned char *nonce, size_t noncelen)
{
    RAND_DRBG_CTR *ctr = &drbg->data.ctr;
    int outlen = AES_BLOCK_SIZE;
    unsigned char V_tmp[48], out[48];
    unsigned char len;

    /* correct key is already set up. */
    memcpy(V_tmp, ctr->V, 16);
    inc_128(ctr);
    memcpy(V_tmp + 16, ctr->V, 16);
    if (ctr->keylen == 16) {
        len = 32;
    } else {
        inc_128(ctr);
        memcpy(V_tmp + 32, ctr->V, 16);
        len = 48;
    }
    if (!EVP_CipherUpdate(ctr->ctx_ecb, out, &outlen, V_tmp, len)
            || outlen != len)
        return 0;
    memcpy(ctr->K, out, ctr->keylen);
    memcpy(ctr->V, out + ctr->keylen, 16);

    if ((drbg->flags & RAND_DRBG_FLAG_CTR_NO_DF) == 0) {
        /* If no input reuse existing derived value */
        if (in1 != NULL || nonce != NULL || in2 != NULL)
            if (!ctr_df(ctr, in1, in1len, nonce, noncelen, in2, in2len))
                return 0;
        /* If this a reuse input in1len != 0 */
        if (in1len)
            ctr_XOR(ctr, ctr->KX, drbg->seedlen);
    } else {
        ctr_XOR(ctr, in1, in1len);
        ctr_XOR(ctr, in2, in2len);
    }

    if (!EVP_CipherInit_ex(ctr->ctx_ecb, NULL, NULL, ctr->K, NULL, -1)
        || !EVP_CipherInit_ex(ctr->ctx_ctr, NULL, NULL, ctr->K, NULL, -1))
        return 0;
    return 1;
}

__owur static int drbg_ctr_instantiate(RAND_DRBG *drbg,
                                       const unsigned char *entropy, size_t entropylen,
                                       const unsigned char *nonce, size_t noncelen,
                                       const unsigned char *pers, size_t perslen)
{
    RAND_DRBG_CTR *ctr = &drbg->data.ctr;

    if (entropy == NULL)
        return 0;

    memset(ctr->K, 0, sizeof(ctr->K));
    memset(ctr->V, 0, sizeof(ctr->V));
    if (!EVP_CipherInit_ex(ctr->ctx_ecb, NULL, NULL, ctr->K, NULL, -1))
        return 0;

    inc_128(ctr);
    if (!ctr_update(drbg, entropy, entropylen, pers, perslen, nonce, noncelen))
        return 0;
    return 1;
}

__owur static int drbg_ctr_reseed(RAND_DRBG *drbg,
                                  const unsigned char *entropy, size_t entropylen,
                                  const unsigned char *adin, size_t adinlen)
{
    RAND_DRBG_CTR *ctr = &drbg->data.ctr;

    if (entropy == NULL)
        return 0;

    inc_128(ctr);
    if (!ctr_update(drbg, entropy, entropylen, adin, adinlen, NULL, 0))
        return 0;
    return 1;
}

static void ctr96_inc(unsigned char *counter)
{
    u32 n = 12, c = 1;

    do {
        --n;
        c += counter[n];
        counter[n] = (u8)c;
        c >>= 8;
    } while (n);
}

__owur static int drbg_ctr_generate(RAND_DRBG *drbg,
                                    unsigned char *out, size_t outlen,
                                    const unsigned char *adin, size_t adinlen)
{
    RAND_DRBG_CTR *ctr = &drbg->data.ctr;
    unsigned int ctr32, blocks;
    int outl, buflen;

    if (adin != NULL && adinlen != 0) {
        inc_128(ctr);

        if (!ctr_update(drbg, adin, adinlen, NULL, 0, NULL, 0))
            return 0;
        /* This means we reuse derived value */
        if ((drbg->flags & RAND_DRBG_FLAG_CTR_NO_DF) == 0) {
            adin = NULL;
            adinlen = 1;
        }
    } else {
        adinlen = 0;
    }

    inc_128(ctr);

    if (outlen == 0) {
        inc_128(ctr);

        if (!ctr_update(drbg, adin, adinlen, NULL, 0, NULL, 0))
            return 0;
        return 1;
    }

    memset(out, 0, outlen);
    __msan_unpoison(ctr->V, 16 * sizeof(char));

    do {
        if (!EVP_CipherInit_ex(ctr->ctx_ctr,
                               NULL, NULL, NULL, ctr->V, -1))
            return 0;

        /*-
         * outlen has type size_t while EVP_CipherUpdate takes an
         * int argument and thus cannot be guaranteed to process more
         * than 2^31-1 bytes at a time. We process such huge generate
         * requests in 2^30 byte chunks, which is the greatest multiple
         * of AES block size lower than or equal to 2^31-1.
         */
        buflen = outlen > (1U << 30) ? (1U << 30) : outlen;
        blocks = (buflen + 15) / 16;

        ctr32 = GETU32(ctr->V + 12) + blocks;
        if (ctr32 < blocks) {
            /* 32-bit counter overflow into V. */
            if (ctr32 != 0) {
                blocks -= ctr32;
                buflen = blocks * 16;
                ctr32 = 0;
            }
            ctr96_inc(ctr->V);
        }
        PUTU32(ctr->V + 12, ctr32);

        if (!EVP_CipherUpdate(ctr->ctx_ctr, out, &outl, out, buflen)
            || outl != buflen)
            return 0;

        out += buflen;
        outlen -= buflen;
    } while (outlen);

    if (!ctr_update(drbg, adin, adinlen, NULL, 0, NULL, 0))
        return 0;
    return 1;
}

static int drbg_ctr_uninstantiate(RAND_DRBG *drbg)
{
    EVP_CIPHER_CTX_free(drbg->data.ctr.ctx_ecb);
    EVP_CIPHER_CTX_free(drbg->data.ctr.ctx_ctr);
    EVP_CIPHER_CTX_free(drbg->data.ctr.ctx_df);
    OPENSSL_cleanse(&drbg->data.ctr, sizeof(drbg->data.ctr));
    return 1;
}

static RAND_DRBG_METHOD drbg_ctr_meth = {
    drbg_ctr_instantiate,
    drbg_ctr_reseed,
    drbg_ctr_generate,
    drbg_ctr_uninstantiate
};

int drbg_ctr_init(RAND_DRBG *drbg)
{
    RAND_DRBG_CTR *ctr = &drbg->data.ctr;
    size_t keylen;

    switch (drbg->type) {
    default:
        /* This can't happen, but silence the compiler warning. */
        return 0;
    case NID_aes_128_ctr:
        keylen = 16;
        ctr->cipher_ecb = EVP_aes_128_ecb();
        ctr->cipher_ctr = EVP_aes_128_ctr();
        break;
    case NID_aes_192_ctr:
        keylen = 24;
        ctr->cipher_ecb = EVP_aes_192_ecb();
        ctr->cipher_ctr = EVP_aes_192_ctr();
        break;
    case NID_aes_256_ctr:
        keylen = 32;
        ctr->cipher_ecb = EVP_aes_256_ecb();
        ctr->cipher_ctr = EVP_aes_256_ctr();
        break;
    }

    drbg->meth = &drbg_ctr_meth;

    ctr->keylen = keylen;
    if (ctr->ctx_ecb == NULL)
        ctr->ctx_ecb = EVP_CIPHER_CTX_new();
    if (ctr->ctx_ctr == NULL)
        ctr->ctx_ctr = EVP_CIPHER_CTX_new();
    if (ctr->ctx_ecb == NULL || ctr->ctx_ctr == NULL
        || !EVP_CipherInit_ex(ctr->ctx_ecb,
                              ctr->cipher_ecb, NULL, NULL, NULL, 1)
        || !EVP_CipherInit_ex(ctr->ctx_ctr,
                              ctr->cipher_ctr, NULL, NULL, NULL, 1))
        return 0;

    drbg->meth = &drbg_ctr_meth;
    drbg->strength = keylen * 8;
    drbg->seedlen = keylen + 16;

    if ((drbg->flags & RAND_DRBG_FLAG_CTR_NO_DF) == 0) {
        /* df initialisation */
        static const unsigned char df_key[32] = {
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
            0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f
        };

        if (ctr->ctx_df == NULL)
            ctr->ctx_df = EVP_CIPHER_CTX_new();
        if (ctr->ctx_df == NULL)
            return 0;
        /* Set key schedule for df_key */
        if (!EVP_CipherInit_ex(ctr->ctx_df,
                               ctr->cipher_ecb, NULL, df_key, NULL, 1))
            return 0;

        drbg->min_entropylen = ctr->keylen;
        drbg->max_entropylen = DRBG_MAX_LENGTH;
        drbg->min_noncelen = drbg->min_entropylen / 2;
        drbg->max_noncelen = DRBG_MAX_LENGTH;
        drbg->max_perslen = DRBG_MAX_LENGTH;
        drbg->max_adinlen = DRBG_MAX_LENGTH;
    } else {
        drbg->min_entropylen = drbg->seedlen;
        drbg->max_entropylen = drbg->seedlen;
        /* Nonce not used */
        drbg->min_noncelen = 0;
        drbg->max_noncelen = 0;
        drbg->max_perslen = drbg->seedlen;
        drbg->max_adinlen = drbg->seedlen;
    }

    drbg->max_request = 1 << 16;

    return 1;
}
