## Math library.

This library contains code for transcendental math. It completely avoids any use of doubles with
the consequential limits on precision. Its purpose is speed not precision. If you need better
precision use another libary. For example the C++ std library (which falls back to slow double
emulation on machines without hardware double support).

All the code in this library is derived from https://netlib.org/cephes under the license
terms given in this file: https://netlib.org/cephes/readme reproduced below:

```
   Some software in this archive may be from the book _Methods and
Programs for Mathematical Functions_ (Prentice-Hall or Simon & Schuster
International, 1989) or from the Cephes Mathematical Library, a
commercial product. In either event, it is copyrighted by the author.
What you see here may be used freely but it comes with no support or
guarantee.

   The two known misprints in the book are repaired here in the
source listings for the gamma function and the incomplete beta
integral.


   Stephen L. Moshier
   moshier@na-net.ornl.gov

```
