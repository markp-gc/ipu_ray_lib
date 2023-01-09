#pragma once

// Efficient sincos that avoids use of double precision.
int sincos(float x, float& s, float& c, int flg=0);
