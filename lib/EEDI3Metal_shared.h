#ifndef EEDI3METAL_SHARED_H
#define EEDI3METAL_SHARED_H

struct EEDI3Params {
    int width;
    int height;
    int mdis;
    int tpitch; // Derived: 2 * mdis + 1
    float alpha, beta, gamma;
    float remainingWeight; // Derived: 1.0 - alpha - beta
    int cost3;
    int ucubic;
    int has_mclip;
    int nrad;
    int field;
    int stride;
    int dh;
};

#endif