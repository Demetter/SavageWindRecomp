#pragma once
#include <stdint.h>
#include <string.h>

// Visit exactly the original sampler's words and tail bytes. Independent chains
// let the CPU overlap loads/multiplies; this does not reduce change-detection
// coverage or alter the separate full-content verification policy.
static uint64_t texture_sample_hash4(const uint8_t *p, uint64_t n) {
    uint64_t h0=1469598103934665603ull, h1=0x9e3779b97f4a7c15ull;
    uint64_t h2=0xc2b2ae3d27d4eb4full, h3=0x165667b19e3779f9ull;
    uint64_t words=n/8, step=words>2048 ? words/2048 : 1, i=0;
#define MIX(h, at) do { uint64_t w; memcpy(&w,p+(at)*8,8); (h)=((h)^w)*1099511628211ull; (h)^=(h)>>29; } while(0)
    for (; i<words && words-i>3*step; i+=4*step) {
        MIX(h0,i); MIX(h1,i+step); MIX(h2,i+2*step); MIX(h3,i+3*step);
    }
    if(i<words) { MIX(h0,i); i+=step; }
    if(i<words) { MIX(h1,i); i+=step; }
    if(i<words) { MIX(h2,i); }
#undef MIX
    uint64_t h=h0 ^ h1*0x9e3779b97f4a7c15ull ^ h2*0xc2b2ae3d27d4eb4full ^ h3*0x165667b19e3779f9ull;
    for(uint64_t j=words*8;j<n;j++) { h^=p[j]; h*=1099511628211ull; }
    return h ^ n;
}
