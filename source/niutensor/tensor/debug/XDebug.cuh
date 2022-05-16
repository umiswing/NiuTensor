#include <XDevice.h>
#include <XTensor.h>
#include <fstream>
namespace nts {
enum EqualErrorKind { EEK_Equal, EEK_Order, EEK_Type, EEK_Dim, EEK_Value };
void _printTensor(const XTensor &X, const int &Dim, const int &Offset);
void printTensor(const XTensor &X);
template <typename T>
inline EqualErrorKind _equal(const XTensor &LHS, const XTensor &RHS) {
  if (LHS.order != RHS.order)
    return EEK_Order;

  if (LHS.dataType != RHS.dataType)
    return EEK_Type;

  for (auto i = 0; i < LHS.order; ++i) {
    if (LHS.GetDim(i) != RHS.GetDim(i))
      return EEK_Dim;
  }
  T *LAddress = (T *)malloc(LHS.unitNum * sizeof(T));
  T *RAddress = (T *)malloc(RHS.unitNum * sizeof(T));
  cudaMemcpy(LAddress, LHS.data, LHS.unitNum * sizeof(T),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(RAddress, RHS.data, RHS.unitNum * sizeof(T),
             cudaMemcpyDeviceToHost);

  for (auto i = 0; i < LHS.unitNum; ++i) {
    if (LAddress[i] != RAddress[i]) {
      free(LAddress);
      free(RAddress);
      return EEK_Value;
    }
  }
  free(LAddress);
  free(RAddress);
  return EEK_Equal;
}
inline EqualErrorKind equal(const XTensor &LHS, const XTensor &RHS) {
  if (LHS.dataType == X_INT)
    return _equal<int>(LHS, RHS);
  else if (LHS.dataType == X_INT8)
    return _equal<int8_t>(LHS, RHS);
  else if (LHS.dataType == X_FLOAT)
    return _equal<float>(LHS, RHS);
  else if (LHS.dataType == X_DOUBLE)
    return _equal<double>(LHS, RHS);
  else if (LHS.dataType == X_FLOAT16) {
    ShowNTErrors("XEqual:XEqual on X_FLOAT is not supported yet!");
  } else {
    ShowNTErrors("XEqual:Unknown data type");
  }
}
template <typename T>
inline void _writeXTensor(XTensor const &X, std::string const &FileName) {
  CheckNTErrors(X.isInit, "writeXTensor():Tensor is not initialized!");
  std::ofstream Out(FileName);
  if (X.devID != -1) {
    T *HostData = (T *)malloc(X.unitNum * sizeof(T));
    int DevIDBackup = 0;
    ProtectCudaDev(X.devID, DevIDBackup);
    cudaMemcpy(HostData, X.data, X.unitNum * sizeof(T), cudaMemcpyDeviceToHost);
    BacktoCudaDev(X.devID, DevIDBackup);
    auto i = 0;
    Out << HostData[i++];
    for (; i < X.unitNum; ++i)
      Out << '\n' << HostData[i];
    free(HostData);
    Out.close();
    return;
  }
  auto i = 0;
  T *Address = (T *)X.data;
  Out << Address[i++];
  for (; i < X.unitNum; ++i)
    Out << '\n' << Address[i];
  Out.close();
  return;
}
inline void writeXTensor(XTensor const &X, std::string const FileName) {
  if (X.dataType == X_INT)
    return _writeXTensor<int>(X, FileName);
  else if (X.dataType == X_INT8)
    return _writeXTensor<int8_t>(X, FileName);
  else if (X.dataType == X_FLOAT)
    return _writeXTensor<float>(X, FileName);
  else if (X.dataType == X_DOUBLE)
    return _writeXTensor<double>(X, FileName);
  else if (X.dataType == X_FLOAT16) {
    ShowNTErrors("writeXTensor(): X_FLOAT16 is not supported yet!");
  } else {
    ShowNTErrors("writeXTensor(): Unknown data type");
  }
}

inline void writeDimSize(XTensor const &X, std::string const &FileName) {
  CheckNTErrors(X.isInit, "writeDimSize():Tensor is not initialized!");
  std::ofstream Out(FileName,std::ios_base::app);
  for (auto i = 0; i < X.order; ++i)
    Out << X.dimSize[i] << '\n';
  Out.close();
  return;
}
} // namespace nts