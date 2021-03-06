/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2017, Natural Language Processing Lab, Northeastern University. 
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-09-25
 */

#ifndef __RESHAPE_H__
#define __RESHAPE_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* reshape the tensor */
XTensor Reshape(XTensor &s, int order, int * dimSize, bool inplace);

void Reshape(XTensor &s, XTensor &t, int order, int * dimSize);

} // namespace nts(NiuTrans.Tensor)
#endif // __RESHAPE_H__
