#include <chrono>
#include <gmpxx.h>
#include <iostream>
using namespace std;

const int N = 1048576; //模数为p，数组长度限制为N

const mpz_class P("4179340454199820289");

mpz_class A[N];

int r[N];

void qpow(mpz_class &res, const mpz_class &x, int y, const mpz_class &p) {

  res = 1;

  mpz_class base = x;

  while (y) {

    if (y & 1) {

      res = res * base % p;
    }

    base = base * base % p;

    y >>= 1;
  }
}

void ntt(mpz_class *x, int lim, int opt, const mpz_class &p) {

  int i, j, k, m;

  mpz_class gn, g, tmp;

  for (i = 0; i < lim; ++i)

    if (r[i] < i)

      swap(x[i], x[r[i]]);

  for (m = 2; m <= lim; m <<= 1) {

    k = m >> 1;

    mpz_class exp = (p - 1) / m;

    qpow(gn, 3, exp.get_ui(), p);

    for (i = 0; i < lim; i += m) {

      g = 1;

      for (j = 0; j < k; ++j) {

        tmp = x[i + j + k] * g % p;

        x[i + j + k] = (x[i + j] - tmp + p) % p;

        x[i + j] = (x[i + j] + tmp) % p;

        g = g * gn % p;
      }
    }
  }
}

int main() {

  srand(time(nullptr));

  int i, lim = 1, n = N / 2; // 设 n 是 N 的一半，确保进行乘法后数据位数不会溢出

  // 用随机数填充 A 和 B，用大数组模拟十进制下的大数表示

  for (i = 0; i < n; i++) {

    A[i] = rand() % 10;
  }

  // 计算适当的 lim 值

  while (lim < n)
    lim <<= 1;

  // 初始化 r 数组

  for (i = 0; i < lim; ++i) {

    r[i] = (i & 1) * (lim >> 1) + (r[i >> 1] >> 1);
  }

  auto start = chrono::high_resolution_clock::now();

  // 对 A 和 B 进行 NTT

  ntt(A, lim, 1, P);

  auto end = chrono::high_resolution_clock::now();

  chrono::duration<double, nano> elapsed = end - start;

  cout << "baseline_cpu " << elapsed.count() / 1000000
       << " ms \n"; // 计算两次NTT的总时间，单位为纳秒
  return 0;
}
