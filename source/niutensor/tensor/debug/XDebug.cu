#include "XDebug.cuh"
#include <iostream>
namespace nts {
void _printTensor(const XTensor &X, const int &Dim, const int &Offset) {
  if (Dim == X.order - 1) {
    const int Cols = X.GetDim(X.order - 1);
    const DTYPE *Address = (DTYPE *)X.data + Offset;
    std::cout << "[";
    size_t Col;
    for (Col = 0; Col < Cols - 1; ++Col) {
      std::cout << Address[Col] << ",";
    }
    std::cout << Address[Col] << "]";
    return;
  }
  int Stride = 1;
  for (size_t i = 0; i < X.order; ++i) {
    if (i > Dim)
      Stride *= X.GetDim(i);
  }
  std::cout << "[";
  for (size_t i = 0; i < X.GetDim(Dim); ++i) {
    _printTensor(X, Dim + 1, Offset + i * Stride);
    if (X.GetDim(Dim) - 1 == i)
      std::cout << "]";
    else
      std::cout << ",";
  }
}

void printTensor(const XTensor &X) {
  _printTensor(X, 0, 0);
  std::cout << "\n";
}
} // namespace nts
